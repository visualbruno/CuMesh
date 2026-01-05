from typing import *
import torch
from tqdm import tqdm
from . import _C
from .bvh import cuBVH


def _init_hashmap(resolution, capacity, device):
    VOL = resolution * resolution * resolution
    if VOL < 2**32:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2**64:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Volume {VOL} > 2^64.")

    hashmap_vals = torch.empty((capacity,), dtype=torch.uint32, device=device)
    return hashmap_keys, hashmap_vals


def get_morton_order(coords: torch.Tensor):
    """
    Interleave bits of x, y, z coordinates to create a Morton code for spatial locality.
    """
    c = coords.long()
    def spread_bits(x):
        x = (x | (x << 16)) & 0x030000FF0000FF
        x = (x | (x << 8))  & 0x0300F00F00F00F
        x = (x | (x << 4))  & 0x030C30C30C30C3
        x = (x | (x << 2))  & 0x09249249249249
        return x
    morton = (spread_bits(c[:, 0]) << 2) | (spread_bits(c[:, 1]) << 1) | spread_bits(c[:, 2])
    return torch.argsort(morton)


def remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    resolution: int,
    band: float = 1,
    project_back: float = 0,
    verbose: bool = False,
    bvh = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized Remeshing using Morton-coded chunks, Dual Contouring, 
    Laplacian relaxation, and spatial projection.
    """
    device = vertices.device
    
    # -------------------------------------------------------------------------
    # 1. Constants & Lazy Initialization
    # -------------------------------------------------------------------------
    if not hasattr(remesh_narrow_band_dc, "edge_neighbor_voxel_offset"):
        remesh_narrow_band_dc.edge_neighbor_voxel_offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],     # x-axis
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],     # y-axis
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],     # z-axis
        ], dtype=torch.int32).unsqueeze(0).to(device)

    # Triangle split tables
    s1_n, s1_p = torch.tensor([0, 1, 2, 0, 2, 3], device=device), torch.tensor([0, 2, 1, 0, 3, 2], device=device)
    s2_n, s2_p = torch.tensor([0, 1, 3, 3, 1, 2], device=device), torch.tensor([0, 3, 1, 3, 2, 1], device=device)

    OFFSETS = torch.tensor([
        [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]
    ], dtype=torch.int32, device=device)

    # -------------------------------------------------------------------------
    # 2. Helpers: Morton Sorting & Chunked Distance
    # -------------------------------------------------------------------------
    def get_morton_indices(coords: torch.Tensor):
        c = coords.long()
        def spread(x):
            x = (x | (x << 16)) & 0x030000FF0000FF
            x = (x | (x << 8))  & 0x0300F00F00F00F
            x = (x | (x << 4))  & 0x030C30C30C30C3
            x = (x | (x << 2))  & 0x09249249249249
            return x
        morton = (spread(c[:, 0]) << 2) | (spread(c[:, 1]) << 1) | spread(c[:, 2])
        return torch.argsort(morton)

    def chunked_udf(bvh_ptr, pts, chunk_size=1048576, return_uvw=False):
        N = pts.shape[0]
        distances = torch.empty(N, dtype=torch.float32, device=device)
        face_ids = torch.empty(N, dtype=torch.int64, device=device)
        uvws = torch.empty((N, 3), dtype=torch.float32, device=device) if return_uvw else None
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            d, f, u = bvh_ptr.unsigned_distance(pts[i:end], return_uvw=return_uvw)
            distances[i:end], face_ids[i:end] = d, f
            if return_uvw: uvws[i:end] = u
        return distances, face_ids, uvws

    # -------------------------------------------------------------------------
    # 3. Sparse Grid Construction (Z-Order Optimized)
    # -------------------------------------------------------------------------
    if bvh is None:
        if verbose: print("Building BVH...")
        bvh = cuBVH(vertices, faces)
    
    eps = band * scale / resolution
    base_res = resolution
    while base_res > 32: base_res //= 2
    
    # 1. Initial coarse grid
    coords = torch.stack(torch.meshgrid(
        torch.arange(base_res, device=device),
        torch.arange(base_res, device=device),
        torch.arange(base_res, device=device),
        indexing='ij'
    ), dim=-1).int().reshape(-1, 3)
    
    # Apply Morton sort to initial grid
    coords = coords[get_morton_indices(coords)]

    pbar = tqdm(total=int(torch.log2(torch.tensor(resolution // base_res)).item()) + 1, 
                desc="Sparse Grid", disable=not verbose)

    while True:
        # Update progress bar at start of iteration
        pbar.update(1)
        
        cell_size = scale / base_res
        pts = ((coords.float() + 0.5) / base_res - 0.5) * scale + center
        
        # Optimized chunked distance with spatial locality
        dist, _, _ = chunked_udf(bvh, pts)
        coords = coords[torch.abs(dist - eps) < 0.87 * cell_size]
        
        if base_res >= resolution:
            break
            
        base_res *= 2
        coords = (coords.unsqueeze(1) * 2 + OFFSETS.unsqueeze(0)).reshape(-1, 3)
        
        # Re-sort after subdivision to maintain spatial locality for next UDF call
        coords = coords[get_morton_indices(coords)]
        
    pbar.close()

    # -------------------------------------------------------------------------
    # 4. Dual Contouring Kernels
    # -------------------------------------------------------------------------
    Nvox = coords.shape[0]
    hashmap_vox = _init_hashmap(resolution, 2 * Nvox, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vox, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1), resolution, resolution, resolution)
    
    grid_verts = _C.get_sparse_voxel_grid_active_vertices(*hashmap_vox, coords.contiguous(), resolution, resolution, resolution)
    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
    dist_vert, _, _ = chunked_udf(bvh, pts_vert)
    
    hashmap_vert = _init_hashmap(resolution + 1, 2 * grid_verts.shape[0], device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vert, torch.cat([torch.zeros_like(grid_verts[:, :1]), grid_verts], dim=1), resolution+1, resolution+1, resolution+1)
    dual_verts, intersected = _C.simple_dual_contour(*hashmap_vert, coords, dist_vert - eps, resolution+1, resolution+1, resolution+1)

    # -------------------------------------------------------------------------
    # 5. Topology (Chunked for Memory)
    # -------------------------------------------------------------------------
    quad_list, dir_list, chunk_size = [], [], 1048576
    non_zero = torch.nonzero(intersected)
    for i in range(0, non_zero.shape[0], chunk_size):
        chunk = non_zero[i:i+chunk_size]
        v_idx, a_idx = chunk[:, 0], chunk[:, 1]
        neighs = coords[v_idx].unsqueeze(1) + remesh_narrow_band_dc.edge_neighbor_voxel_offset[0, a_idx]
        keys = torch.cat([torch.zeros((neighs.shape[0]*4, 1), dtype=torch.int, device=device), neighs.reshape(-1, 3)], dim=1)
        lookup = _C.hashmap_lookup_3d_cuda(*hashmap_vox, keys, resolution, resolution, resolution).reshape(-1, 4).int()
        mask = (lookup != 0xffffffff).all(dim=1)
        if mask.any():
            quad_list.append(lookup[mask]); dir_list.append(intersected[v_idx[mask], a_idx[mask]])

    if not quad_list: return torch.zeros((0,3), device=device), torch.zeros((0,3), device=device, dtype=torch.int32)
    quad_indices = torch.cat(quad_list, dim=0)
    intersected_dir = torch.cat(dir_list, dim=0).int()

    # Re-index to remove unused vertices
    unique_v, inv_idx = torch.unique(quad_indices, return_inverse=True)
    dual_verts = dual_verts[unique_v]
    quad_indices = inv_idx.reshape(-1, 4).int()
    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center

    # -------------------------------------------------------------------------
    # 6. Smoothing & Detail Projection
    # -------------------------------------------------------------------------
    if project_back > 0:
        if verbose: print("Relaxing & Projecting...")
        # Laplacian Smoothing for topological quality
        row, col = quad_indices[:, [0,1,2,3]].flatten(), quad_indices[:, [1,2,3,0]].flatten()
        adj = torch.sparse_coo_tensor(torch.stack([torch.cat([row,col]), torch.cat([col,row])]), torch.ones(row.shape[0]*2, device=device), (mesh_vertices.shape[0], mesh_vertices.shape[0]))
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1).unsqueeze(-1)
        for _ in range(3): mesh_vertices = torch.sparse.mm(adj, mesh_vertices) / deg
        
        # Spatially sorted projection for BVH cache locality
        v_c = ((mesh_vertices - center) / scale + 0.5) * resolution
        sort_idx = get_morton_indices(v_c.clamp(0, resolution - 1))
        inv_sort = torch.empty_like(sort_idx); inv_sort[sort_idx] = torch.arange(sort_idx.size(0), device=device)
        
        _, f_id_s, uvw_s = chunked_udf(bvh, mesh_vertices[sort_idx], return_uvw=True)
        proj_v = (vertices[faces[f_id_s[inv_sort].long()]] * uvw_s[inv_sort].unsqueeze(-1)).sum(dim=1)
        mesh_vertices = mesh_vertices + project_back * (proj_v - mesh_vertices)

    # -------------------------------------------------------------------------
    # 7. Final Triangle Construction
    # -------------------------------------------------------------------------
    mesh_triangles = torch.empty((quad_indices.shape[0] * 2, 3), dtype=torch.int32, device=device)
    for i in range(0, quad_indices.shape[0], chunk_size):
        end = min(i + chunk_size, quad_indices.shape[0])
        q, d = quad_indices[i:end], intersected_dir[i:end].unsqueeze(1)
        
        # Test Split 1
        t1 = torch.where(d == 1, q[:, s1_p], q[:, s1_n])
        n1_a = torch.cross(mesh_vertices[t1[:,1]]-mesh_vertices[t1[:,0]], mesh_vertices[t1[:,2]]-mesh_vertices[t1[:,0]])
        n1_b = torch.cross(mesh_vertices[t1[:,4]]-mesh_vertices[t1[:,3]], mesh_vertices[t1[:,5]]-mesh_vertices[t1[:,3]])
        align1 = (n1_a * n1_b).sum(dim=1).abs()
        
        # Test Split 2
        t2 = torch.where(d == 1, q[:, s2_p], q[:, s2_n])
        n2_a = torch.cross(mesh_vertices[t2[:,1]]-mesh_vertices[t2[:,0]], mesh_vertices[t2[:,2]]-mesh_vertices[t2[:,0]])
        n2_b = torch.cross(mesh_vertices[t2[:,4]]-mesh_vertices[t2[:,3]], mesh_vertices[t2[:,5]]-mesh_vertices[t2[:,3]])
        align2 = (n2_a * n2_b).sum(dim=1).abs()
        
        mesh_triangles[i*2:end*2] = torch.where((align1 > align2).unsqueeze(1), t1, t2).reshape(-1, 3)

    return mesh_vertices, mesh_triangles