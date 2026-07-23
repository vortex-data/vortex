// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// CUB DeviceSelect::Flagged wrapper for Vortex GPU filtering.

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

// i256 type
typedef struct {
    __int128_t high;
    __int128_t low;
} __int256_t;

// Bit extraction functor for TransformInputIterator
struct BitExtractor {
    const uint8_t *packed;
    uint64_t bit_offset;

    __host__ __device__ inline uint8_t operator()(int64_t idx) const {
        uint64_t actual_bit = bit_offset + static_cast<uint64_t>(idx);
        uint64_t byte_idx = actual_bit / 8;
        uint32_t bit_idx = actual_bit % 8;
        return (packed[byte_idx] >> bit_idx) & 1;
    }
};

/// Type alias for the packed bit iterator.
using PackedBitIterator = thrust::transform_iterator<BitExtractor, thrust::counting_iterator<int64_t>>;

// CUB DeviceSelect::Flagged - Query temp storage size
template <typename T>
static cudaError_t filter_temp_size_impl(size_t *temp_bytes, int64_t num_items) {
    size_t bytes = 0;
    cudaError_t err = cub::DeviceSelect::Flagged(nullptr,
                                                 bytes,
                                                 static_cast<const T *>(nullptr),
                                                 static_cast<const uint8_t *>(nullptr),
                                                 static_cast<T *>(nullptr),
                                                 static_cast<int64_t *>(nullptr),
                                                 num_items);
    *temp_bytes = bytes;
    return err;
}

#define DEFINE_TEMP_SIZE(suffix, Type)                                                                       \
    extern "C" cudaError_t filter_temp_size_##suffix(size_t *temp_bytes, int64_t n) {                        \
        return filter_temp_size_impl<Type>(temp_bytes, n);                                                   \
    }

DEFINE_TEMP_SIZE(u8, uint8_t)
DEFINE_TEMP_SIZE(i8, int8_t)
DEFINE_TEMP_SIZE(u16, uint16_t)
DEFINE_TEMP_SIZE(i16, int16_t)
DEFINE_TEMP_SIZE(u32, uint32_t)
DEFINE_TEMP_SIZE(i32, int32_t)
DEFINE_TEMP_SIZE(u64, uint64_t)
DEFINE_TEMP_SIZE(i64, int64_t)
DEFINE_TEMP_SIZE(f32, float)
DEFINE_TEMP_SIZE(f64, double)
DEFINE_TEMP_SIZE(i128, __int128_t)
DEFINE_TEMP_SIZE(i256, __int256_t)

// CUB DeviceSelect::Flagged - Execute filter with byte mask (one byte per element)
template <typename T>
static cudaError_t filter_bytemask_impl(void *d_temp,
                                        size_t temp_bytes,
                                        const T *d_in,
                                        const uint8_t *d_flags,
                                        T *d_out,
                                        int64_t *d_num_selected,
                                        int64_t num_items,
                                        cudaStream_t stream) {
    return cub::DeviceSelect::Flagged(d_temp,
                                      temp_bytes,
                                      d_in,
                                      d_flags,
                                      d_out,
                                      d_num_selected,
                                      num_items,
                                      stream);
}

#define DEFINE_FILTER_BYTEMASK(suffix, Type)                                                                 \
    extern "C" cudaError_t filter_bytemask_##suffix(void *d_temp,                                            \
                                                    size_t temp_bytes,                                       \
                                                    const Type *d_in,                                        \
                                                    const uint8_t *d_flags,                                  \
                                                    Type *d_out,                                             \
                                                    int64_t *d_num_selected,                                 \
                                                    int64_t num_items,                                       \
                                                    cudaStream_t stream) {                                   \
        return filter_bytemask_impl<Type>(d_temp,                                                            \
                                          temp_bytes,                                                        \
                                          d_in,                                                              \
                                          d_flags,                                                           \
                                          d_out,                                                             \
                                          d_num_selected,                                                    \
                                          num_items,                                                         \
                                          stream);                                                           \
    }

DEFINE_FILTER_BYTEMASK(u8, uint8_t)
DEFINE_FILTER_BYTEMASK(i8, int8_t)
DEFINE_FILTER_BYTEMASK(u16, uint16_t)
DEFINE_FILTER_BYTEMASK(i16, int16_t)
DEFINE_FILTER_BYTEMASK(u32, uint32_t)
DEFINE_FILTER_BYTEMASK(i32, int32_t)
DEFINE_FILTER_BYTEMASK(u64, uint64_t)
DEFINE_FILTER_BYTEMASK(i64, int64_t)
DEFINE_FILTER_BYTEMASK(f32, float)
DEFINE_FILTER_BYTEMASK(f64, double)
DEFINE_FILTER_BYTEMASK(i128, __int128_t)
DEFINE_FILTER_BYTEMASK(i256, __int256_t)

/// CUB DeviceSelect::Flagged - Execute filter with bit mask (one bit per element)
///
/// Execute filter is using packed bit mask directly via TransformInputIterator.
///
/// # Parameters
///
/// * `d_temp` - Temporary storage buffer
/// * `temp_bytes` - Size of temporary storage
/// * `d_in` - Input data array
/// * `d_bitmask` - Packed bit mask (1 bit per element)
/// * `bit_offset` - Starting bit offset within the packed buffer
/// * `d_out` - Output array for selected elements
/// * `d_num_selected` - Output count of selected elements
/// * `num_items` - Number of input elements
/// * `stream` - CUDA stream
template <typename T>
static cudaError_t filter_bitmask_impl(void *d_temp,
                                       size_t temp_bytes,
                                       const T *d_in,
                                       const uint8_t *d_bitmask,
                                       uint64_t bit_offset,
                                       T *d_out,
                                       int64_t *d_num_selected,
                                       int64_t num_items,
                                       cudaStream_t stream) {
    BitExtractor extractor {d_bitmask, bit_offset};
    thrust::counting_iterator<int64_t> counting_iter(0);
    PackedBitIterator flag_iter(counting_iter, extractor);

    return cub::DeviceSelect::Flagged(d_temp,
                                      temp_bytes,
                                      d_in,
                                      flag_iter,
                                      d_out,
                                      d_num_selected,
                                      num_items,
                                      stream);
}

#define DEFINE_FILTER_BITMASK(suffix, Type)                                                                  \
    extern "C" cudaError_t filter_bitmask_##suffix(void *d_temp,                                             \
                                                   size_t temp_bytes,                                        \
                                                   const Type *d_in,                                         \
                                                   const uint8_t *d_bitmask,                                 \
                                                   uint64_t bit_offset,                                      \
                                                   Type *d_out,                                              \
                                                   int64_t *d_num_selected,                                  \
                                                   int64_t num_items,                                        \
                                                   cudaStream_t stream) {                                    \
        return filter_bitmask_impl<Type>(d_temp,                                                             \
                                         temp_bytes,                                                         \
                                         d_in,                                                               \
                                         d_bitmask,                                                          \
                                         bit_offset,                                                         \
                                         d_out,                                                              \
                                         d_num_selected,                                                     \
                                         num_items,                                                          \
                                         stream);                                                            \
    }

DEFINE_FILTER_BITMASK(u8, uint8_t)
DEFINE_FILTER_BITMASK(i8, int8_t)
DEFINE_FILTER_BITMASK(u16, uint16_t)
DEFINE_FILTER_BITMASK(i16, int16_t)
DEFINE_FILTER_BITMASK(u32, uint32_t)
DEFINE_FILTER_BITMASK(i32, int32_t)
DEFINE_FILTER_BITMASK(u64, uint64_t)
DEFINE_FILTER_BITMASK(i64, int64_t)
DEFINE_FILTER_BITMASK(f32, float)
DEFINE_FILTER_BITMASK(f64, double)
DEFINE_FILTER_BITMASK(i128, __int128_t)
DEFINE_FILTER_BITMASK(i256, __int256_t)

// Query CUB temporary storage for an exclusive-sum scan.
template <typename T>
static cudaError_t scan_exclusive_sum_temp_size_impl(size_t *temp_bytes, int64_t num_items) {
    size_t bytes = 0;
    cudaError_t err = cub::DeviceScan::ExclusiveSum(nullptr,
                                                    bytes,
                                                    static_cast<const T *>(nullptr),
                                                    static_cast<T *>(nullptr),
                                                    num_items);
    *temp_bytes = bytes;
    return err;
}

// Export one temp-size query and scan launch per supported element type.
#define DEFINE_SCAN_EXCLUSIVE_SUM(suffix, Type)                                                              \
    extern "C" cudaError_t scan_exclusive_sum_##suffix##_temp_size(size_t *temp_bytes, int64_t num_items) {  \
        return scan_exclusive_sum_temp_size_impl<Type>(temp_bytes, num_items);                               \
    }                                                                                                        \
    extern "C" cudaError_t scan_exclusive_sum_##suffix(void *d_temp,                                         \
                                                       size_t temp_bytes,                                    \
                                                       const Type *d_in,                                     \
                                                       Type *d_out,                                          \
                                                       int64_t num_items,                                    \
                                                       cudaStream_t stream) {                                \
        return cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, num_items, stream);            \
    }

DEFINE_SCAN_EXCLUSIVE_SUM(i32, int32_t)
DEFINE_SCAN_EXCLUSIVE_SUM(i64, int64_t)

// Widening length functor for TransformInputIterator: index `i < num_rows`
// reads `lengths[i]` widened to u64; the final index contributes zero, so an
// exclusive scan over `num_rows + 1` items ends with the total. A negative
// length (signed types only) raises `*status` to 2 and contributes zero.
template <typename LenT>
struct WideningLength {
    const LenT *lengths;
    int64_t num_rows;
    uint32_t *status;

    __host__ __device__ inline uint64_t operator()(int64_t idx) const {
        if (idx >= num_rows) {
            return 0;
        }
        const LenT len = lengths[idx];
        if constexpr (static_cast<LenT>(-1) < static_cast<LenT>(0)) {
            if (len < static_cast<LenT>(0)) {
#ifdef __CUDA_ARCH__
                atomicMax(status, 2u);
#endif
                return 0;
            }
        }
        return static_cast<uint64_t>(len);
    }
};

template <typename LenT>
using WideningLengthIterator =
    thrust::transform_iterator<WideningLength<LenT>, thrust::counting_iterator<int64_t>>;

// Query CUB temporary storage for the fused widen + exclusive-sum scan.
template <typename LenT>
static cudaError_t scan_exclusive_sum_lengths_temp_size_impl(size_t *temp_bytes,
                                                             int64_t num_offsets) {
    WideningLengthIterator<LenT> lengths_it(thrust::counting_iterator<int64_t>(0),
                                            WideningLength<LenT>{nullptr, 0, nullptr});
    size_t bytes = 0;
    cudaError_t err = cub::DeviceScan::ExclusiveSum(nullptr,
                                                    bytes,
                                                    lengths_it,
                                                    static_cast<uint64_t *>(nullptr),
                                                    num_offsets);
    *temp_bytes = bytes;
    return err;
}

// Fused widen + exclusive-sum: one CUB dispatch scans `num_offsets` values
// where value `i` is `lengths[i]` widened to u64 (zero for the final slot),
// writing u64 offsets whose last element is the total — no materialized
// widened input.
#define DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(suffix, Type)                                                      \
    extern "C" cudaError_t scan_exclusive_sum_lengths_##suffix##_temp_size(size_t *temp_bytes,               \
                                                                           int64_t num_offsets) {            \
        return scan_exclusive_sum_lengths_temp_size_impl<Type>(temp_bytes, num_offsets);                     \
    }                                                                                                        \
    extern "C" cudaError_t scan_exclusive_sum_lengths_##suffix(void *d_temp,                                 \
                                                               size_t temp_bytes,                            \
                                                               const Type *lengths,                          \
                                                               uint64_t *d_out,                              \
                                                               uint32_t *status,                             \
                                                               int64_t num_offsets,                          \
                                                               cudaStream_t stream) {                        \
        WideningLengthIterator<Type> lengths_it(thrust::counting_iterator<int64_t>(0),                       \
                                                WideningLength<Type>{lengths, num_offsets - 1, status});     \
        return cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, lengths_it, d_out, num_offsets, stream);    \
    }

DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(u8, uint8_t)
DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(i8, int8_t)
DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(u16, uint16_t)
DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(i16, int16_t)
DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(u32, uint32_t)
DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(i32, int32_t)
DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(u64, uint64_t)
DEFINE_SCAN_EXCLUSIVE_SUM_LENGTHS(i64, int64_t)
