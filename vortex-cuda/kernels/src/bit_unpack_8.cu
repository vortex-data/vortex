// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_8_lanes.cuh"
#include "patches.cuh"

template <int BW>
__device__ void _bit_unpack_8_device(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, int thread_idx, GPUPatches& patches) {
    __shared__ uint8_t shared_out[1024];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        _bit_unpack_8_lane<BW>(in, shared_out, reference, thread_idx * 4 + i);
    }
    __syncwarp();

    // Step 2: Apply patches to shared memory in parallel
    PatchesCursor<uint8_t> cursor(patches, blockIdx.x, thread_idx, 32);
    auto patch = cursor.next();
    while (patch.index != 1024) {
        shared_out[patch.index] = patch.value;
        patch = cursor.next();
    }
    __syncwarp();

    // Step 3: Copy to global memory
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_8_0bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<0>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_1bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<1>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_2bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<2>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_3bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<3>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_4bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<4>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_5bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<5>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_6bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<6>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_7bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<7>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_8_8bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_device<8>(in, out, reference, thread_idx, patches);
}

