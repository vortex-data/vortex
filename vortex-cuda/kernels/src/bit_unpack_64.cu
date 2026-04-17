// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_64_lanes.cuh"
#include "patches.cuh"

template <int BW>
__device__ void _bit_unpack_64_device(const uint64_t *__restrict in, uint64_t *__restrict out, uint64_t reference, int thread_idx, GPUPatches& patches) {
    __shared__ uint64_t shared_out[1024];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < FL_LANES<uint64_t> / 16; i++) {
        _bit_unpack_64_lane<BW>(in, shared_out, reference, thread_idx * (FL_LANES<uint64_t> / 16) + i);
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
    for (int i = 0; i < FL_CHUNK / 16; i++) {
        auto idx = i * 16 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_64_0bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 0));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<0>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_1bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 1));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<1>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_2bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 2));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<2>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_3bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 3));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<3>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_4bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 4));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<4>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_5bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 5));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<5>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_6bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 6));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<6>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_7bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 7));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<7>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_8bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 8));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<8>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_9bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 9));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<9>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_10bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 10));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<10>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_11bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 11));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<11>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_12bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 12));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<12>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_13bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 13));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<13>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_14bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 14));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<14>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_15bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 15));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<15>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_16bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 16));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<16>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_17bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 17));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<17>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_18bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 18));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<18>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_19bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 19));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<19>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_20bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 20));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<20>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_21bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 21));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<21>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_22bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 22));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<22>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_23bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 23));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<23>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_24bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 24));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<24>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_25bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 25));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<25>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_26bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 26));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<26>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_27bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 27));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<27>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_28bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 28));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<28>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_29bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 29));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<29>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_30bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 30));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<30>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_31bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 31));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<31>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_32bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 32));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<32>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_33bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 33));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<33>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_34bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 34));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<34>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_35bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 35));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<35>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_36bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 36));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<36>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_37bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 37));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<37>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_38bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 38));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<38>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_39bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 39));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<39>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_40bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 40));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<40>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_41bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 41));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<41>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_42bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 42));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<42>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_43bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 43));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<43>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_44bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 44));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<44>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_45bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 45));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<45>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_46bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 46));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<46>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_47bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 47));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<47>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_48bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 48));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<48>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_49bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 49));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<49>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_50bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 50));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<50>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_51bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 51));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<51>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_52bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 52));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<52>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_53bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 53));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<53>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_54bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 54));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<54>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_55bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 55));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<55>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_56bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 56));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<56>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_57bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 57));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<57>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_58bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 58));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<58>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_59bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 59));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<59>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_60bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 60));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<60>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_61bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 61));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<61>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_62bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 62));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<62>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_63bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 63));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<63>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_64_64bw_16t(const uint64_t *__restrict full_in, uint64_t *__restrict full_out, uint64_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint64_t> * 64));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_64_device<64>(in, out, reference, thread_idx, patches);
}

