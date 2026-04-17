// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_64_lanes.cuh"
#include "patches.cuh"

template <int BW>
__device__ void _bit_unpack_64_device(const uint64_t *__restrict in, uint64_t *__restrict out, uint64_t reference, int thread_idx, GPUPatches& patches) {
    __shared__ uint64_t shared_out[1024];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < 1; i++) {
        _bit_unpack_64_lane<BW>(in, shared_out, reference, thread_idx * 1 + i);
    }
    __syncwarp();

    // Step 2: Apply patches to shared memory in parallel
    PatchesCursor<uint64_t> cursor(patches, blockIdx.x, thread_idx, 16);
    auto patch = cursor.next();
    while (patch.index != 1024) {
        shared_out[patch.index] = patch.value;
        patch = cursor.next();
    }
    __syncwarp();

    // Step 3: Copy to global memory
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        auto idx = i * 16 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_64_0bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<0>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_1bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<1>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_2bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<2>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_3bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<3>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_4bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<4>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_5bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<5>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_6bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<6>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_7bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<7>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_8bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<8>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_9bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 9 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<9>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_10bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 10 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<10>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_11bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 11 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<11>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_12bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 12 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<12>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_13bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 13 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<13>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_14bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 14 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<14>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_15bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 15 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<15>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_16bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 16 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<16>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_17bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 17 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<17>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_18bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 18 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<18>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_19bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 19 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<19>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_20bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 20 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<20>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_21bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 21 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<21>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_22bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 22 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<22>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_23bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 23 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<23>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_24bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 24 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<24>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_25bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 25 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<25>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_26bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 26 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<26>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_27bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 27 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<27>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_28bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 28 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<28>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_29bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 29 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<29>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_30bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 30 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<30>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_31bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 31 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<31>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_32bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 32 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<32>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_33bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 33 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<33>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_34bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 34 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<34>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_35bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 35 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<35>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_36bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 36 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<36>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_37bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 37 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<37>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_38bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 38 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<38>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_39bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 39 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<39>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_40bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 40 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<40>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_41bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 41 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<41>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_42bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 42 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<42>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_43bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 43 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<43>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_44bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 44 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<44>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_45bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 45 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<45>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_46bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 46 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<46>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_47bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 47 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<47>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_48bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 48 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<48>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_49bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 49 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<49>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_50bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 50 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<50>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_51bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 51 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<51>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_52bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 52 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<52>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_53bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 53 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<53>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_54bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 54 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<54>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_55bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 55 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<55>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_56bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 56 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<56>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_57bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 57 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<57>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_58bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 58 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<58>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_59bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 59 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<59>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_60bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 60 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<60>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_61bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 61 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<61>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_62bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 62 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<62>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_63bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 63 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<63>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_64bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 64 / sizeof(uint64_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_64_device<64>(in, out, reference, thread_idx, patches);
}

