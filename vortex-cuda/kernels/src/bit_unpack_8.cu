// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_8_lanes.cuh"
#include "patches.cuh"

template <int BW>
__device__ void _bit_unpack_8_device(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, int thread_idx, GPUPatches& patches) {
    __shared__ uint8_t shared_out[FL_CHUNK];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < FL_LANES<uint8_t> / 32; i++) {
        _bit_unpack_8_lane<BW>(in, shared_out, reference, thread_idx * (FL_LANES<uint8_t> / 32) + i);
    }
    __syncwarp();

    // Step 2: Apply patches to shared memory in parallel
    PatchesCursor<uint8_t> cursor(patches, blockIdx.x, thread_idx, 32);
    auto patch = cursor.next();
    while (patch.index != FL_CHUNK) {
        shared_out[patch.index] = patch.value;
        patch = cursor.next();
    }
    __syncwarp();

    // Step 3: Copy to global memory
    #pragma unroll
    for (int i = 0; i < FL_CHUNK / 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_8_0bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 0));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<0>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_1bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 1));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<1>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_2bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 2));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<2>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_3bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 3));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<3>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_4bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 4));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<4>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_5bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 5));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<5>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_6bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 6));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<6>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_7bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 7));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<7>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_8bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint8_t> * 8));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_8_device<8>(in, out, reference, thread_idx, patches);
}

