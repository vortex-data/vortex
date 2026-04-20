// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_16_lanes.cuh"
#include "patches.cuh"

template <int BW>
__device__ void _bit_unpack_16_device(const uint16_t *__restrict in, uint16_t *__restrict out, uint16_t reference, int thread_idx, GPUPatches& patches) {
    __shared__ uint16_t shared_out[1024];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        _bit_unpack_16_lane<BW>(in, shared_out, reference, thread_idx * 2 + i);
    }
    __syncwarp();

    // Step 2: Apply patches to shared memory in parallel
    PatchesCursor<uint16_t> cursor(patches, blockIdx.x, thread_idx, 32);
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

extern "C" __global__ void bit_unpack_16_0bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<0>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_1bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<1>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_2bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<2>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_3bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<3>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_4bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<4>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_5bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<5>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_6bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<6>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_7bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<7>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_8bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<8>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_9bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 9 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<9>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_10bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 10 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<10>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_11bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 11 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<11>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_12bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 12 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<12>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_13bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 13 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<13>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_14bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 14 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<14>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_15bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 15 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<15>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_16bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 16 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_device<16>(in, out, reference, thread_idx, patches);
}

