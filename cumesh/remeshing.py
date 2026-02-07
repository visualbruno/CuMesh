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


def reconstruct_mesh_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int = 256,
    band: float = 1,
    verbose: bool = False,
    bvh = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct a mesh using dual contouring to obtain correct face normals.
    
    This function uses a voxel-based approach similar to remesh_narrow_band_dc,
    but without smoothing or projection steps. The resulting mesh will have
    consistent outward-facing normals determined by the SDF sign transitions.
    
    Note: This reconstructs the mesh geometry, so there will be slight differences
    from the original due to voxelization. For higher fidelity, use higher resolution.
    
    Args:
        vertices: [V, 3] tensor of vertex positions
        faces: [F, 3] tensor of face indices
        resolution: Voxel grid resolution (higher = more detail, more memory)
        band: Narrow band width in voxel units (default 1)
        verbose: Print progress information
        bvh: Optional pre-built cuBVH
        
    Returns:
        Tuple of (new_vertices, new_faces) with correct normals
    """
    device = vertices.device
    
    # Compute bounding box and scale
    bbox_min = vertices.min(dim=0).values
    bbox_max = vertices.max(dim=0).values
    center = (bbox_min + bbox_max) / 2
    scale = (bbox_max - bbox_min).max().item() * 1.1  # 10% padding
    
    # --- 1. Constants ---
    edge_neighbor_voxel_offset = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=torch.int32, device=device).unsqueeze(0)

    s1_n, s1_p = torch.tensor([0, 1, 2, 0, 2, 3], device=device), torch.tensor([0, 2, 1, 0, 3, 2], device=device)
    s2_n, s2_p = torch.tensor([0, 1, 3, 3, 1, 2], device=device), torch.tensor([0, 3, 1, 3, 2, 1], device=device)
    OFFSETS = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=torch.int32, device=device)

    # --- 2. Helpers ---
    def get_morton_indices(coords: torch.Tensor):
        if coords.shape[0] > 10_000_000:
            return torch.arange(coords.shape[0], device=device)
        c = coords.long()
        def spread(x):
            x = (x | (x << 16)) & 0x030000FF0000FF
            x = (x | (x << 8))  & 0x0300F00F00F00F
            x = (x | (x << 4))  & 0x030C30C30C30C3
            x = (x | (x << 2))  & 0x09249249249249
            return x
        morton = (spread(c[:, 0]) << 2) | (spread(c[:, 1]) << 1) | spread(c[:, 2])
        return torch.argsort(morton)

    def chunked_udf(bvh_ptr, pts, chunk_size=524288):
        N = pts.shape[0]
        distances = torch.empty(N, dtype=torch.float32, device=device)
        face_ids = torch.empty(N, dtype=torch.int64, device=device)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            d, f, _ = bvh_ptr.unsigned_distance(pts[i:end])
            distances[i:end], face_ids[i:end] = d, f
        return distances, face_ids

    # --- 3. Sparse Grid Construction ---
    if bvh is None: 
        bvh = cuBVH(vertices.detach().clone(), faces.detach().clone())
    eps = max(band * scale / resolution, scale / resolution * 0.5) 
    base_res = resolution
    while base_res > 32: base_res //= 2
    
    coords = torch.stack(torch.meshgrid(
        torch.arange(base_res, device=device), 
        torch.arange(base_res, device=device), 
        torch.arange(base_res, device=device), 
        indexing='ij'
    ), dim=-1).int().reshape(-1, 3)
    
    pbar = tqdm(total=int(torch.log2(torch.tensor(resolution // base_res)).item()) + 1, desc="Sparse Grid", disable=not verbose)
    while True:
        pbar.update(1)
        cell_size = scale / base_res
        pts = ((coords.float() + 0.5) / base_res - 0.5) * scale + center
        dist, _ = chunked_udf(bvh, pts)
        coords = coords[torch.abs(dist - eps) < 0.87 * cell_size + eps * 0.5]
        if base_res >= resolution: break
        base_res *= 2
        # Chunked voxel expansion to avoid OOM on large grids
        expand_chunk_size = 250_000  # Process 250K coords at a time
        if coords.shape[0] <= expand_chunk_size:
            coords = (coords.unsqueeze(1) * 2 + OFFSETS.unsqueeze(0)).reshape(-1, 3)
        else:
            # Process in chunks with progressive deduplication
            result_coords = None
            for i in range(0, coords.shape[0], expand_chunk_size):
                end = min(i + expand_chunk_size, coords.shape[0])
                chunk = coords[i:end]
                expanded = (chunk.unsqueeze(1) * 2 + OFFSETS.unsqueeze(0)).reshape(-1, 3)
                if result_coords is None:
                    result_coords = expanded
                else:
                    result_coords = torch.cat([result_coords, expanded], dim=0)
                del expanded
                # Deduplicate every 4 chunks to keep memory bounded
                if (i // expand_chunk_size + 1) % 4 == 0 or end >= coords.shape[0]:
                    result_coords = torch.unique(result_coords, dim=0)
                    torch.cuda.empty_cache() if device.type == 'cuda' else None
            coords = result_coords
            del result_coords
        if coords.shape[0] < 8_000_000:
            coords = coords[get_morton_indices(coords)]
    pbar.close()

    # --- 4. Dual Contouring ---
    if verbose:
        pbar_dc = tqdm(total=5, desc="Dual Contouring")
    Nvox = coords.shape[0]
    if Nvox == 0:
        if verbose:
            pbar_dc.close()
        return torch.zeros((0,3), device=device), torch.zeros((0,3), device=device, dtype=torch.int32)
    
    hashmap_vox = _init_hashmap(resolution, 2 * Nvox, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vox, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1), resolution, resolution, resolution)
    if verbose:
        pbar_dc.update(1)  # Voxel hashmap
    grid_verts = _C.get_sparse_voxel_grid_active_vertices(*hashmap_vox, coords.contiguous(), resolution, resolution, resolution)
    if verbose:
        pbar_dc.update(1)  # Grid vertices
    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
    dist_vert, _ = chunked_udf(bvh, pts_vert)
    if verbose:
        pbar_dc.update(1)  # UDF computation
    hashmap_vert = _init_hashmap(resolution + 1, 2 * grid_verts.shape[0], device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vert, torch.cat([torch.zeros_like(grid_verts[:, :1]), grid_verts], dim=1), resolution+1, resolution+1, resolution+1)
    if verbose:
        pbar_dc.update(1)  # Vertex hashmap
    dual_verts, intersected = _C.simple_dual_contour(*hashmap_vert, coords, dist_vert - eps, resolution+1, resolution+1, resolution+1)
    if verbose:
        pbar_dc.update(1)  # Dual contour
        pbar_dc.close()     

    # --- 5. Topology ---
    quad_list, dir_list, chunk_size = [], [], 524288
    non_zero = torch.nonzero(intersected)
    num_chunks = (non_zero.shape[0] + chunk_size - 1) // chunk_size
    pbar_topo = tqdm(total=num_chunks, desc="Building Topology", disable=not verbose)
    for i in range(0, non_zero.shape[0], chunk_size):
        chunk = non_zero[i:i+chunk_size]
        v_idx, a_idx = chunk[:, 0], chunk[:, 1]
        neighs = coords[v_idx].unsqueeze(1) + edge_neighbor_voxel_offset[0, a_idx]
        keys = torch.cat([torch.zeros((neighs.shape[0]*4, 1), dtype=torch.int, device=device), neighs.reshape(-1, 3)], dim=1)
        lookup = _C.hashmap_lookup_3d_cuda(*hashmap_vox, keys, resolution, resolution, resolution).reshape(-1, 4).int()
        mask = (lookup != 0xffffffff).all(dim=1)
        if mask.any():
            quad_list.append(lookup[mask])
            dir_list.append(intersected[v_idx[mask], a_idx[mask]])
        pbar_topo.update(1)
    pbar_topo.close()

    if not quad_list: 
        return torch.zeros((0,3), device=device), torch.zeros((0,3), device=device, dtype=torch.int32)
    
    quad_indices = torch.cat(quad_list, dim=0)
    intersected_dir = torch.cat(dir_list, dim=0).int()

    # Re-indexing
    active_mask = torch.zeros(Nvox, dtype=torch.bool, device=device)
    active_mask[quad_indices.flatten().long()] = True
    vert_map = torch.full((Nvox,), -1, dtype=torch.int32, device=device)
    vert_map[active_mask] = torch.arange(active_mask.sum().item(), dtype=torch.int32, device=device)
    dual_verts = dual_verts[active_mask]
    for i in range(0, quad_indices.shape[0], chunk_size):
        end = min(i + chunk_size, quad_indices.shape[0])
        quad_indices[i:end] = vert_map[quad_indices[i:end].long()]
    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center

    # --- 6. Triangulation with correct normals ---
    mesh_triangles = torch.empty((quad_indices.shape[0] * 2, 3), dtype=torch.int32, device=device)
    num_tri_chunks = (quad_indices.shape[0] + chunk_size - 1) // chunk_size
    pbar_tri = tqdm(total=num_tri_chunks, desc="Triangulation", disable=not verbose)
    for i in range(0, quad_indices.shape[0], chunk_size):
        end = min(i + chunk_size, quad_indices.shape[0])
        q, d = quad_indices[i:end], intersected_dir[i:end].unsqueeze(1)
        t1 = torch.where(d == 1, q[:, s1_p], q[:, s1_n])
        t2 = torch.where(d == 1, q[:, s2_p], q[:, s2_n])
        # Choose split based on normal alignment
        n1_a = torch.cross(mesh_vertices[t1[:,1].long()]-mesh_vertices[t1[:,0].long()], mesh_vertices[t1[:,2].long()]-mesh_vertices[t1[:,0].long()])
        n1_b = torch.cross(mesh_vertices[t1[:,4].long()]-mesh_vertices[t1[:,3].long()], mesh_vertices[t1[:,5].long()]-mesh_vertices[t1[:,3].long()])
        align1 = (n1_a * n1_b).sum(dim=1).abs()
        n2_a = torch.cross(mesh_vertices[t2[:,1].long()]-mesh_vertices[t2[:,0].long()], mesh_vertices[t2[:,2].long()]-mesh_vertices[t2[:,0].long()])
        n2_b = torch.cross(mesh_vertices[t2[:,4].long()]-mesh_vertices[t2[:,3].long()], mesh_vertices[t2[:,5].long()]-mesh_vertices[t2[:,3].long()])
        align2 = (n2_a * n2_b).sum(dim=1).abs()
        mesh_triangles[i*2:end*2] = torch.where((align1 < align2).unsqueeze(1), t1, t2).reshape(-1, 3)
        pbar_tri.update(1)
    pbar_tri.close()

    # --- 7. Remove inner layer ---
    # Use ray stabbing to determine if face centers are inside or outside the original mesh
    # Outer layer: face centers are OUTSIDE original mesh (positive signed distance)
    # Inner layer: face centers are INSIDE original mesh (negative signed distance)
    print('Removing Inner Layer ...')
    # --- 8. Remove inner layer ---
    # Use ray stabbing to determine if face centers are inside or outside the original mesh
    # Outer layer: face centers are OUTSIDE original mesh (positive signed distance)
    # Inner layer: face centers are INSIDE original mesh (negative signed distance)
    face_centers = torch.empty((mesh_triangles.shape[0], 3), dtype=torch.float32, device=device)
    num_chunks = (mesh_triangles.shape[0] + chunk_size - 1) // chunk_size
    pbar_centers = tqdm(total=num_chunks, desc="Computing Face Centers", disable=not verbose)
    for i in range(0, mesh_triangles.shape[0], chunk_size):
        end = min(i + chunk_size, mesh_triangles.shape[0])
        tri_chunk = mesh_triangles[i:end].long()
        # Direct calculation - more memory efficient
        face_centers[i:end] = (
            mesh_vertices[tri_chunk[:, 0]] + 
            mesh_vertices[tri_chunk[:, 1]] + 
            mesh_vertices[tri_chunk[:, 2]]
        ) / 3
        pbar_centers.update(1)
    pbar_centers.close()

    #if verbose:
    #    pbar_layer = tqdm(total=3, desc="Removing Inner Layer")
    #v0 = mesh_vertices[mesh_triangles[:, 0].long()]
    #v1 = mesh_vertices[mesh_triangles[:, 1].long()]
    #v2 = mesh_vertices[mesh_triangles[:, 2].long()]
    #face_centers = (v0 + v1 + v2) / 3
    #if verbose:
    #    pbar_layer.update(1)  # Face centers computed
    
    # Use ray stabbing mode for signed distance (works even with broken normals)
    def chunked_sdf_raystab(bvh_ptr, pts):
        N = pts.shape[0]
        distances = torch.empty(N, dtype=torch.float32, device=device)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            d, _, _ = bvh_ptr.signed_distance(pts[i:end], mode='raystab')
            distances[i:end] = d
        return distances
    
    print('Signed Distance Computing ...')
    signed_dist = chunked_sdf_raystab(bvh, face_centers)
    #if verbose:
    #    pbar_layer.update(1)  # Signed distance computed
    
    # Keep only faces where center is OUTSIDE original mesh (positive signed distance)
    # or very close to surface (within eps tolerance)
    is_outer = signed_dist >= -eps * 0.1
    
    print('Filtering triangles...')
    # Use integer indexing instead of boolean indexing - much faster for large tensors
    # torch.nonzero finds the indices where is_outer is True, then we use integer indexing
    outer_indices = torch.nonzero(is_outer, as_tuple=False).squeeze(1)
    if outer_indices.numel() > 0:
        mesh_triangles = mesh_triangles[outer_indices]
    else:
        mesh_triangles = torch.zeros((0, 3), dtype=torch.int32, device=device)
    print(f'Kept {outer_indices.numel()} outer faces out of {is_outer.numel()} total faces')
    #if verbose:
    #    pbar_layer.update(1)  # Filtering done
    #    pbar_layer.close()
    
    # Re-index vertices to remove unused ones (chunked for performance)
    print('Re-indexing vertices ...')
    used_verts = torch.zeros(mesh_vertices.shape[0], dtype=torch.bool, device=device)
    
    # Mark used vertices in chunks to avoid large flatten() operation
    reindex_chunk_size = 524288
    num_reindex_chunks = (mesh_triangles.shape[0] + reindex_chunk_size - 1) // reindex_chunk_size
    pbar_reindex = tqdm(total=num_reindex_chunks, desc="Marking Used Vertices", disable=not verbose)
    for i in range(0, mesh_triangles.shape[0], reindex_chunk_size):
        end = min(i + reindex_chunk_size, mesh_triangles.shape[0])
        tri_chunk = mesh_triangles[i:end].long()
        # Mark vertices used in this chunk (all 3 vertices per triangle)
        used_verts[tri_chunk[:, 0]] = True
        used_verts[tri_chunk[:, 1]] = True
        used_verts[tri_chunk[:, 2]] = True
        pbar_reindex.update(1)
    pbar_reindex.close()
    
    # Create compact vertex mapping
    new_vert_idx = torch.full((mesh_vertices.shape[0],), -1, dtype=torch.int32, device=device)
    new_vert_idx[used_verts] = torch.arange(used_verts.sum().item(), dtype=torch.int32, device=device)
    mesh_vertices = mesh_vertices[used_verts]
    
    # Remap triangle indices in chunks
    pbar_remap = tqdm(total=num_reindex_chunks, desc="Remapping Triangle Indices", disable=not verbose)
    for i in range(0, mesh_triangles.shape[0], reindex_chunk_size):
        end = min(i + reindex_chunk_size, mesh_triangles.shape[0])
        mesh_triangles[i:end] = new_vert_idx[mesh_triangles[i:end].long()]
        pbar_remap.update(1)
    pbar_remap.close()
    print('Inner Layer removed')

    return mesh_vertices, mesh_triangles


def remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    resolution: int,
    band: float = 1,
    project_back: float = 0,
    verbose: bool = False,
    bvh = None,
    remove_inner_faces: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    device = vertices.device
    
    # --- 1. Constants ---
    # Always create on the current device - don't cache across calls to avoid device mismatch
    edge_neighbor_voxel_offset = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=torch.int32, device=device).unsqueeze(0)

    s1_n, s1_p = torch.tensor([0, 1, 2, 0, 2, 3], device=device), torch.tensor([0, 2, 1, 0, 3, 2], device=device)
    s2_n, s2_p = torch.tensor([0, 1, 3, 3, 1, 2], device=device), torch.tensor([0, 3, 1, 3, 2, 1], device=device)
    OFFSETS = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=torch.int32, device=device)

    # --- 2. Memory-Safe Helpers ---
    def get_morton_indices(coords: torch.Tensor):
        if coords.shape[0] > 10_000_000: # Skip sorting if too large to avoid argsort OOM
            return torch.arange(coords.shape[0], device=device)
        c = coords.long()
        def spread(x):
            x = (x | (x << 16)) & 0x030000FF0000FF
            x = (x | (x << 8))  & 0x0300F00F00F00F
            x = (x | (x << 4))  & 0x030C30C30C30C3
            x = (x | (x << 2))  & 0x09249249249249
            return x
        morton = (spread(c[:, 0]) << 2) | (spread(c[:, 1]) << 1) | spread(c[:, 2])
        return torch.argsort(morton)

    def chunked_udf(bvh_ptr, pts, chunk_size=524288, return_uvw=False):
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

    # --- 3. Sparse Grid Construction ---
    if bvh is None: bvh = cuBVH(vertices.detach().clone(), faces.detach().clone())
    eps = max(band * scale / resolution, scale / resolution * 0.5)
    base_res = resolution
    while base_res > 32: base_res //= 2
    
    coords = torch.stack(torch.meshgrid(torch.arange(base_res, device=device), torch.arange(base_res, device=device), torch.arange(base_res, device=device), indexing='ij'), dim=-1).int().reshape(-1, 3)
    
    pbar = tqdm(total=int(torch.log2(torch.tensor(resolution // base_res)).item()) + 1, desc="Sparse Grid", disable=not verbose)
    while True:
        pbar.update(1)
        cell_size = scale / base_res
        pts = ((coords.float() + 0.5) / base_res - 0.5) * scale + center
        dist, _, _ = chunked_udf(bvh, pts)
        coords = coords[torch.abs(dist - eps) < 0.87 * cell_size + eps * 0.5]
        if base_res >= resolution: break
        base_res *= 2
        # Chunked voxel expansion to avoid OOM on large grids
        expand_chunk_size = 250_000  # Process 250K coords at a time
        if coords.shape[0] <= expand_chunk_size:
            coords = (coords.unsqueeze(1) * 2 + OFFSETS.unsqueeze(0)).reshape(-1, 3)
        else:
            # Process in chunks with progressive deduplication
            result_coords = None
            for i in range(0, coords.shape[0], expand_chunk_size):
                end = min(i + expand_chunk_size, coords.shape[0])
                chunk = coords[i:end]
                expanded = (chunk.unsqueeze(1) * 2 + OFFSETS.unsqueeze(0)).reshape(-1, 3)
                if result_coords is None:
                    result_coords = expanded
                else:
                    result_coords = torch.cat([result_coords, expanded], dim=0)
                del expanded
                # Deduplicate every 4 chunks to keep memory bounded
                if (i // expand_chunk_size + 1) % 4 == 0 or end >= coords.shape[0]:
                    result_coords = torch.unique(result_coords, dim=0)
                    torch.cuda.empty_cache() if device.type == 'cuda' else None
            coords = result_coords
            del result_coords
        # Spatial sorting is most important at the final high-res steps
        if coords.shape[0] < 8_000_000:
            coords = coords[get_morton_indices(coords)]
    pbar.close()

    # --- 4. Dual Contouring Kernels ---
    Nvox = coords.shape[0]
    hashmap_vox = _init_hashmap(resolution, 2 * Nvox, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vox, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1), resolution, resolution, resolution)
    grid_verts = _C.get_sparse_voxel_grid_active_vertices(*hashmap_vox, coords.contiguous(), resolution, resolution, resolution)
    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
    dist_vert, _, _ = chunked_udf(bvh, pts_vert)
    hashmap_vert = _init_hashmap(resolution + 1, 2 * grid_verts.shape[0], device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vert, torch.cat([torch.zeros_like(grid_verts[:, :1]), grid_verts], dim=1), resolution+1, resolution+1, resolution+1)
    dual_verts, intersected = _C.simple_dual_contour(*hashmap_vert, coords, dist_vert - eps, resolution+1, resolution+1, resolution+1)

    # --- 5. Topology (Memory Safe Re-indexing) ---
    quad_list, dir_list, chunk_size = [], [], 524288
    non_zero = torch.nonzero(intersected)
    for i in range(0, non_zero.shape[0], chunk_size):
        chunk = non_zero[i:i+chunk_size]
        v_idx, a_idx = chunk[:, 0], chunk[:, 1]
        neighs = coords[v_idx].unsqueeze(1) + edge_neighbor_voxel_offset[0, a_idx]
        keys = torch.cat([torch.zeros((neighs.shape[0]*4, 1), dtype=torch.int, device=device), neighs.reshape(-1, 3)], dim=1)
        lookup = _C.hashmap_lookup_3d_cuda(*hashmap_vox, keys, resolution, resolution, resolution).reshape(-1, 4).int()
        mask = (lookup != 0xffffffff).all(dim=1)
        if mask.any():
            quad_list.append(lookup[mask]); dir_list.append(intersected[v_idx[mask], a_idx[mask]])

    if not quad_list: return torch.zeros((0,3), device=device), torch.zeros((0,3), device=device, dtype=torch.int32)
    quad_indices = torch.cat(quad_list, dim=0)
    intersected_dir = torch.cat(dir_list, dim=0).int()

    # Re-indexing without torch.unique to save memory
    active_mask = torch.zeros(Nvox, dtype=torch.bool, device=device)
    active_mask[quad_indices.flatten().long()] = True
    vert_map = torch.full((Nvox,), -1, dtype=torch.int32, device=device)
    vert_map[active_mask] = torch.arange(active_mask.sum().item(), dtype=torch.int32, device=device)
    dual_verts = dual_verts[active_mask]
    for i in range(0, quad_indices.shape[0], chunk_size):
        end = min(i + chunk_size, quad_indices.shape[0])
        quad_indices[i:end] = vert_map[quad_indices[i:end].long()]
    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center

    # -------------------------------------------------------------------------
    # 6. Memory-Safe Smoothing & Detail Projection
    # -------------------------------------------------------------------------
    if project_back > 0:
        if verbose: print("Relaxing topology (Chunked)...")
        
        # We use a chunked approach to calculate neighbor averages without a sparse matrix
        num_v = mesh_vertices.shape[0]
        for _ in range(3): # 3 Iterations of smoothing
            v_sum = torch.zeros_like(mesh_vertices)
            v_count = torch.zeros((num_v, 1), device=device)
            
            # Process triangles in chunks to accumulate neighbor positions
            for i in range(0, quad_indices.shape[0], chunk_size):
                end = min(i + chunk_size, quad_indices.shape[0])
                q = quad_indices[i:end].long()
                
                # Each vertex in a quad has two neighbors along the quad edges
                # We accumulate their positions to calculate the local average
                for idx, next_idx in [(0,1), (1,2), (2,3), (3,0)]:
                    v_sum.index_add_(0, q[:, idx], mesh_vertices[q[:, next_idx]])
                    v_sum.index_add_(0, q[:, next_idx], mesh_vertices[q[:, idx]])
                    v_count.index_add_(0, q[:, idx], torch.ones((q.shape[0], 1), device=device))
                    v_count.index_add_(0, q[:, next_idx], torch.ones((q.shape[0], 1), device=device))
            
            # Update positions: Move toward average of neighbors
            mesh_vertices = v_sum / v_count.clamp(min=1)

        # --- Optimized Project Back ---
        if verbose: print("Projecting back to original mesh...")
        v_c = ((mesh_vertices - center) / scale + 0.5) * resolution
        # Sort only if manageable
        if v_c.shape[0] < 8_000_000:
            sort_idx = get_morton_indices(v_c.clamp(0, resolution - 1))
            inv_sort = torch.empty_like(sort_idx); inv_sort[sort_idx] = torch.arange(sort_idx.size(0), device=device)
            _, f_id_s, uvw_s = chunked_udf(bvh, mesh_vertices[sort_idx], return_uvw=True)
            proj_v = (vertices[faces[f_id_s[inv_sort].long()]] * uvw_s[inv_sort].unsqueeze(-1)).sum(dim=1)
        else:
            _, f_id, uvw = chunked_udf(bvh, mesh_vertices, return_uvw=True)
            proj_v = (vertices[faces[f_id.long()]] * uvw.unsqueeze(-1)).sum(dim=1)
            
        mesh_vertices = mesh_vertices + project_back * (proj_v - mesh_vertices)

    # --- 7. Triangulation ---
    mesh_triangles = torch.empty((quad_indices.shape[0] * 2, 3), dtype=torch.int32, device=device)
    for i in range(0, quad_indices.shape[0], chunk_size):
        end = min(i + chunk_size, quad_indices.shape[0])
        q, d = quad_indices[i:end], intersected_dir[i:end].unsqueeze(1)
        t1 = torch.where(d == 1, q[:, s1_p], q[:, s1_n])
        t2 = torch.where(d == 1, q[:, s2_p], q[:, s2_n])
        # Choose split based on normal alignment
        n1_a = torch.cross(mesh_vertices[t1[:,1].long()]-mesh_vertices[t1[:,0].long()], mesh_vertices[t1[:,2].long()]-mesh_vertices[t1[:,0].long()])
        n1_b = torch.cross(mesh_vertices[t1[:,4].long()]-mesh_vertices[t1[:,3].long()], mesh_vertices[t1[:,5].long()]-mesh_vertices[t1[:,3].long()])
        align1 = (n1_a * n1_b).sum(dim=1).abs()
        n2_a = torch.cross(mesh_vertices[t2[:,1].long()]-mesh_vertices[t2[:,0].long()], mesh_vertices[t2[:,2].long()]-mesh_vertices[t2[:,0].long()])
        n2_b = torch.cross(mesh_vertices[t2[:,4].long()]-mesh_vertices[t2[:,3].long()], mesh_vertices[t2[:,5].long()]-mesh_vertices[t2[:,3].long()])
        align2 = (n2_a * n2_b).sum(dim=1).abs()
        mesh_triangles[i*2:end*2] = torch.where((align1 < align2).unsqueeze(1), t1, t2).reshape(-1, 3)

    if remove_inner_faces:
        print('Removing Inner Layer ...')
        # --- 8. Remove inner layer ---
        # Use ray stabbing to determine if face centers are inside or outside the original mesh
        # Outer layer: face centers are OUTSIDE original mesh (positive signed distance)
        # Inner layer: face centers are INSIDE original mesh (negative signed distance)
        chunk_size = 524288
        face_centers = torch.empty((mesh_triangles.shape[0], 3), dtype=torch.float32, device=device)
        num_chunks = (mesh_triangles.shape[0] + chunk_size - 1) // chunk_size
        pbar_centers = tqdm(total=num_chunks, desc="Computing Face Centers", disable=not verbose)
        for i in range(0, mesh_triangles.shape[0], chunk_size):
            end = min(i + chunk_size, mesh_triangles.shape[0])
            tri_chunk = mesh_triangles[i:end].long()
            # Direct calculation - more memory efficient
            face_centers[i:end] = (
                mesh_vertices[tri_chunk[:, 0]] + 
                mesh_vertices[tri_chunk[:, 1]] + 
                mesh_vertices[tri_chunk[:, 2]]
            ) / 3
            pbar_centers.update(1)
        pbar_centers.close()

        #if verbose:
        #    pbar_layer = tqdm(total=3, desc="Removing Inner Layer")
        #v0 = mesh_vertices[mesh_triangles[:, 0].long()]
        #v1 = mesh_vertices[mesh_triangles[:, 1].long()]
        #v2 = mesh_vertices[mesh_triangles[:, 2].long()]
        #face_centers = (v0 + v1 + v2) / 3
        #if verbose:
        #    pbar_layer.update(1)  # Face centers computed
        
        # Use ray stabbing mode for signed distance (works even with broken normals)
        def chunked_sdf_raystab(bvh_ptr, pts):
            N = pts.shape[0]
            distances = torch.empty(N, dtype=torch.float32, device=device)
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                d, _, _ = bvh_ptr.signed_distance(pts[i:end], mode='raystab')
                distances[i:end] = d
            return distances
        
        print('Signed Distance Computing ...')
        signed_dist = chunked_sdf_raystab(bvh, face_centers)
        #if verbose:
        #    pbar_layer.update(1)  # Signed distance computed
        
        # Keep only faces where center is OUTSIDE original mesh (positive signed distance)
        # or very close to surface (within eps tolerance)
        is_outer = signed_dist >= -eps * 0.1
        
        print('Filtering triangles...')
        # Use integer indexing instead of boolean indexing - much faster for large tensors
        # torch.nonzero finds the indices where is_outer is True, then we use integer indexing
        outer_indices = torch.nonzero(is_outer, as_tuple=False).squeeze(1)
        if outer_indices.numel() > 0:
            mesh_triangles = mesh_triangles[outer_indices]
        else:
            mesh_triangles = torch.zeros((0, 3), dtype=torch.int32, device=device)
        print(f'Kept {outer_indices.numel()} outer faces out of {is_outer.numel()} total faces')
        #if verbose:
        #    pbar_layer.update(1)  # Filtering done
        #    pbar_layer.close()
        
        # Re-index vertices to remove unused ones (chunked for performance)
        print('Re-indexing vertices ...')
        used_verts = torch.zeros(mesh_vertices.shape[0], dtype=torch.bool, device=device)
        
        # Mark used vertices in chunks to avoid large flatten() operation
        reindex_chunk_size = 524288
        num_reindex_chunks = (mesh_triangles.shape[0] + reindex_chunk_size - 1) // reindex_chunk_size
        pbar_reindex = tqdm(total=num_reindex_chunks, desc="Marking Used Vertices", disable=not verbose)
        for i in range(0, mesh_triangles.shape[0], reindex_chunk_size):
            end = min(i + reindex_chunk_size, mesh_triangles.shape[0])
            tri_chunk = mesh_triangles[i:end].long()
            # Mark vertices used in this chunk (all 3 vertices per triangle)
            used_verts[tri_chunk[:, 0]] = True
            used_verts[tri_chunk[:, 1]] = True
            used_verts[tri_chunk[:, 2]] = True
            pbar_reindex.update(1)
        pbar_reindex.close()
        
        # Create compact vertex mapping
        new_vert_idx = torch.full((mesh_vertices.shape[0],), -1, dtype=torch.int32, device=device)
        new_vert_idx[used_verts] = torch.arange(used_verts.sum().item(), dtype=torch.int32, device=device)
        mesh_vertices = mesh_vertices[used_verts]
        
        # Remap triangle indices in chunks
        pbar_remap = tqdm(total=num_reindex_chunks, desc="Remapping Triangle Indices", disable=not verbose)
        for i in range(0, mesh_triangles.shape[0], reindex_chunk_size):
            end = min(i + reindex_chunk_size, mesh_triangles.shape[0])
            mesh_triangles[i:end] = new_vert_idx[mesh_triangles[i:end].long()]
            pbar_remap.update(1)
        pbar_remap.close()
        print('Inner Layer removed')

    return mesh_vertices, mesh_triangles