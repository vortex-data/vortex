// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_16_lanes.cuh"
#include "patches.cuh"

template <int BW>
__device__ void _bit_unpack_16_device(const uint16_t *__restrict in, uint16_t *__restrict out, uint16_t reference, int thread_idx, GPUPatches& patches) {
    __shared__ uint16_t shared_out[1024];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < FL_LANES<uint16_t> / 32; i++) {
        _bit_unpack_16_lane<BW>(in, shared_out, reference, thread_idx * (FL_LANES<uint16_t> / 32) + i);
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
    for (int i = 0; i < FL_CHUNK / 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_16_0bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 0));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<0>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_1bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 1));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<1>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_2bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 2));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<2>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_3bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 3));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<3>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_4bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 4));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<4>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_5bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 5));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<5>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_6bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 6));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<6>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_7bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 7));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<7>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_8bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 8));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<8>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_9bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 9));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<9>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_10bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 10));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<10>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_11bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 11));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<11>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_12bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 12));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<12>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_13bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 13));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<13>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_14bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 14));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<14>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_15bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 15));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<15>(in, out, reference, thread_idx, patches);
}

extern "C" __global__ void bit_unpack_16_16bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out, uint16_t reference, GPUPatches patches) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (FL_LANES<uint16_t> * 16));
    auto out = full_out + (blockIdx.x * FL_CHUNK);
    _bit_unpack_16_device<16>(in, out, reference, thread_idx, patches);
}

