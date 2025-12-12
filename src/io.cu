#include "cumesh.h"


namespace cumesh {


static __global__ void copy_array_to_pack4x3(
    const int* in,
    const int num_elements,
    int3* out
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    const int base = 3 * tid;
    out[tid] = { in[base + 0], in[base + 1], in[base + 2] };
}


static __global__ void copy_pack4x3_to_array(
    const int3* in,
    const int num_elements,
    int* out
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    const int base = 3 * tid;
    int3 v = in[tid];
    out[base + 0] = v.x;
    out[base + 1] = v.y;
    out[base + 2] = v.z;
}


void CuMesh::init(const torch::Tensor& vertices, const torch::Tensor& faces) {
    size_t num_vertices = vertices.size(0);
    size_t num_faces = faces.size(0);
    this->vertices.resize(num_vertices);
    this->faces.resize(num_faces);
    copy_array_to_pack4x3<<<(num_vertices+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const int*>(vertices.data_ptr<float>()),
        num_vertices,
        reinterpret_cast<int3*>(this->vertices.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    copy_array_to_pack4x3<<<(num_faces+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        faces.data_ptr<int>(),
        num_faces,
        this->faces.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


std::tuple<torch::Tensor, torch::Tensor> CuMesh::read() {
    auto vertices = torch::empty({ static_cast<int64_t>(this->vertices.size), 3 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto faces = torch::empty({ static_cast<int64_t>(this->faces.size), 3 }, torch::dtype(torch::kInt32).device(torch::kCUDA));

    copy_pack4x3_to_array<<<(this->vertices.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const int3*>(this->vertices.ptr),
        this->vertices.size,
        reinterpret_cast<int*>(vertices.data_ptr<float>())
    );
    CUDA_CHECK(cudaGetLastError());
    copy_pack4x3_to_array<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        this->faces.size,
        faces.data_ptr<int>()
    );
    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(vertices, faces);
}


torch::Tensor CuMesh::read_face_normals() {
    auto face_normals = torch::empty({ static_cast<int64_t>(this->faces.size), 3 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    copy_pack4x3_to_array<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const int3*>(this->face_normals.ptr),
        this->faces.size,
        reinterpret_cast<int*>(face_normals.data_ptr<float>())
    );
    CUDA_CHECK(cudaGetLastError());
    return face_normals;
}


torch::Tensor CuMesh::read_vertex_normals() {
    auto vertex_normals = torch::empty({ static_cast<int64_t>(this->vertex_normals.size), 3 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    copy_pack4x3_to_array<<<(this->vertex_normals.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const int3*>(this->vertex_normals.ptr),
        this->vertex_normals.size,
        reinterpret_cast<int*>(vertex_normals.data_ptr<float>())
    );
    CUDA_CHECK(cudaGetLastError());
    return vertex_normals;
}


torch::Tensor CuMesh::read_edges() {
    auto edges = torch::empty({ static_cast<int64_t>(this->edges.size), 2 }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        edges.data_ptr<int>(),
        this->edges.ptr,
        this->edges.size * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice
    ));
    return edges;
}


torch::Tensor CuMesh::read_boundaries() {
    auto boundaries = torch::empty({ static_cast<int64_t>(this->boundaries.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        boundaries.data_ptr<int>(),
        this->boundaries.ptr,
        this->boundaries.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    return boundaries;
}


torch::Tensor CuMesh::read_manifold_face_adjacency() {
    auto manifold_face_adj = torch::empty({ static_cast<int64_t>(this->manifold_face_adj.size), 2 }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        manifold_face_adj.data_ptr<int>(),
        this->manifold_face_adj.ptr,
        this->manifold_face_adj.size * sizeof(int2),
        cudaMemcpyDeviceToDevice
    ));
    return manifold_face_adj;
}


torch::Tensor CuMesh::read_manifold_boundary_adjacency() {
    auto manifold_bound_adj = torch::empty({ static_cast<int64_t>(this->manifold_bound_adj.size), 2 }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        manifold_bound_adj.data_ptr<int>(),
        this->manifold_bound_adj.ptr,
        this->manifold_bound_adj.size * sizeof(int2),
        cudaMemcpyDeviceToDevice
    ));
    return manifold_bound_adj;
}

  
std::tuple<int, torch::Tensor> CuMesh::read_connected_components() {
    auto conn_comp_ids_tensor = torch::empty({ static_cast<int64_t>(this->conn_comp_ids.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        conn_comp_ids_tensor.data_ptr<int>(),
        this->conn_comp_ids.ptr,
        this->conn_comp_ids.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    return std::make_tuple(this->num_conn_comps, conn_comp_ids_tensor);
}


std::tuple<int, torch::Tensor> CuMesh::read_boundary_connected_components() {
    auto bound_conn_comp_ids_tensor = torch::empty({ static_cast<int64_t>(this->bound_conn_comp_ids.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        bound_conn_comp_ids_tensor.data_ptr<int>(),
        this->bound_conn_comp_ids.ptr,
        this->bound_conn_comp_ids.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    return std::make_tuple(this->num_bound_conn_comps, bound_conn_comp_ids_tensor);
}


std::tuple<int, torch::Tensor, torch::Tensor> CuMesh::read_boundary_loops() {
    auto loop_boundaries_tensor = torch::empty({ static_cast<int64_t>(this->loop_boundaries.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto loop_boundaries_offset_tensor = torch::empty({ static_cast<int64_t>(this->loop_boundaries_offset.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        loop_boundaries_tensor.data_ptr<int>(),
        this->loop_boundaries.ptr,
        this->loop_boundaries.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        loop_boundaries_offset_tensor.data_ptr<int>(),
        this->loop_boundaries_offset.ptr,
        this->loop_boundaries_offset.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    return std::make_tuple(this->num_bound_loops, loop_boundaries_tensor, loop_boundaries_offset_tensor);
}


} // namespace cumesh
