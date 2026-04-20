// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_32_lanes.cuh"
#include "patches.cuh"

template <int BW>
__device__ void _bit_unpack_32_device(const uint32_t *__restrict in, uint32_t *__restrict out, uint32_t reference, int thread_idx, GPUPatches& patches) {
    __shared__ uint32_t shared_out[1024];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < 1; i++) {
        _bit_unpack_32_lane<BW>(in, shared_out, reference, thread_idx * 1 + i);
    }
    __syncwarp();

    // Step 2: Apply patches to shared memory in parallel
    PatchesCursor<uint32_t> cursor(patches, blockIdx.x, thread_idx, 32);
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

extern "C" __global__ void bit_unpack_32_0bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<0>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_1bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<1>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_2bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<2>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_3bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<3>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_4bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<4>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_5bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<5>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_6bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<6>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_7bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<7>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_8bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<8>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_9bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 9 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<9>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_10bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 10 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<10>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_11bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 11 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<11>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_12bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 12 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<12>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_13bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 13 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<13>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_14bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 14 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<14>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_15bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 15 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<15>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_16bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 16 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<16>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_17bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 17 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<17>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_18bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 18 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<18>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_19bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 19 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<19>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_20bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 20 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<20>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_21bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 21 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<21>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_22bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 22 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<22>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_23bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 23 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<23>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_24bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 24 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<24>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_25bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 25 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<25>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_26bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 26 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<26>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_27bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 27 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<27>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_28bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 28 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<28>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_29bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 29 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<29>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_30bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 30 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<30>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_31bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 31 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<31>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_32_32bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out, uint32_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 32 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_device<32>(in, out, reference, thread_idx, patches);
}

