// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

// Keep these layouts and limits in sync with exact_filter.rs. Descriptors are kernel
// parameters, not device allocations: there is no metadata H2D transfer per batch.
constexpr uint32_t FILTER_THREADS = 256;
constexpr uint32_t FILTER_TILE_ROWS = 2048;
constexpr uint32_t FILTER_MAX_PREDICATES = 16;
constexpr uint32_t FILTER_MAX_COLUMNS = 32;

struct FilterComparison {
    uint64_t input;
    uint64_t literal;
    uint32_t ptype;
    uint32_t op;
};
struct FilterPredicates {
    FilterComparison comparisons[FILTER_MAX_PREDICATES];
    uint32_t group_ends[FILTER_MAX_PREDICATES];
};
struct FilterColumn {
    uint64_t input;
    uint64_t output;
    uint64_t width;
};
struct FilterColumns {
    FilterColumn columns[FILTER_MAX_COLUMNS];
};
static_assert(sizeof(FilterComparison) == 24);
static_assert(sizeof(FilterPredicates) == 448);
static_assert(sizeof(FilterColumn) == 24);
static_assert(sizeof(FilterColumns) == 768);

template <typename T>
__device__ bool filter_compare(T value, T literal, uint32_t op) {
    // Match compare.cu: downstream IEEE and Vortex total ordering disagree on NaNs
    // and signed zeros, so ambiguous rows must survive this superset prefilter.
    if constexpr (std::is_floating_point_v<T>) {
        if (isnan(value) || isnan(literal) || (value == 0 && literal == 0)) {
            return true;
        }
    }
    switch (op) {
    case 0: return value == literal;
    case 1: return value != literal;
    case 2: return value > literal;
    case 3: return value >= literal;
    case 4: return value < literal;
    case 5: return value <= literal;
    default: return true;
    }
}

template <typename T>
__device__ T filter_literal(uint64_t bits) {
    return static_cast<T>(bits);
}

template <>
__device__ float filter_literal<float>(uint64_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits));
}

template <>
__device__ double filter_literal<double>(uint64_t bits) {
    return __longlong_as_double(static_cast<long long>(bits));
}

template <typename T>
__device__ bool filter_compare_group(const FilterPredicates &predicates,
                                     uint32_t first,
                                     uint32_t end,
                                     uint64_t row) {
    const T value = reinterpret_cast<const T *>(predicates.comparisons[first].input)[row];
    bool keep = true;
    for (uint32_t index = first; index < end; ++index) {
        const FilterComparison &comparison = predicates.comparisons[index];
        keep = keep && filter_compare(value, filter_literal<T>(comparison.literal), comparison.op);
    }
    return keep;
}

__device__ bool filter_predicate_group(const FilterPredicates &predicates, uint32_t first, uint64_t row) {
    const uint32_t end = predicates.group_ends[first];
    switch (predicates.comparisons[first].ptype) {
    case 0: return filter_compare_group<uint8_t>(predicates, first, end, row);
    case 1: return filter_compare_group<uint16_t>(predicates, first, end, row);
    case 2: return filter_compare_group<uint32_t>(predicates, first, end, row);
    case 3: return filter_compare_group<uint64_t>(predicates, first, end, row);
    case 4: return filter_compare_group<int8_t>(predicates, first, end, row);
    case 5: return filter_compare_group<int16_t>(predicates, first, end, row);
    case 6: return filter_compare_group<int32_t>(predicates, first, end, row);
    case 7: return filter_compare_group<int64_t>(predicates, first, end, row);
    case 8: return filter_compare_group<float>(predicates, first, end, row);
    case 9: return filter_compare_group<double>(predicates, first, end, row);
    default: return true;
    }
}

// All 256 threads participate. Each warp publishes its sum before other warps read it.
// No block ever waits for another block; inter-block ordering is provided by launches.
__device__ uint64_t filter_exclusive_sum(uint64_t value, uint64_t *warp_sums) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    uint64_t inclusive = value;
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
        const uint64_t previous = __shfl_up_sync(0xffffffff, inclusive, offset);
        if (lane >= offset) {
            inclusive += previous;
        }
    }
    if (lane == 31) {
        warp_sums[warp] = inclusive;
    }
    __syncthreads();
    uint64_t prefix = inclusive - value;
    for (uint32_t preceding = 0; preceding < warp; ++preceding) {
        prefix += warp_sums[preceding];
    }
    return prefix;
}

// One bitmap pass for the entire conjunction, also producing the per-tile counts.
extern "C" __global__ void exact_filter_mask(FilterPredicates predicates,
                                              uint32_t num_predicates,
                                              uint64_t len,
                                              uint32_t *__restrict words,
                                              uint64_t *__restrict tile_counts) {
    __shared__ uint64_t warp_sums[FILTER_THREADS / 32];
    const uint64_t tile_start = static_cast<uint64_t>(blockIdx.x) * FILTER_TILE_ROWS;
    const uint32_t lane = threadIdx.x % 32;
    uint64_t count = 0;
    for (uint32_t offset = threadIdx.x; offset < FILTER_TILE_ROWS; offset += FILTER_THREADS) {
        const uint64_t row = tile_start + offset;
        bool keep = row < len;
        for (uint32_t first = 0; first < num_predicates; first = predicates.group_ends[first]) {
            keep = keep && filter_predicate_group(predicates, first, row);
        }
        const uint32_t bits = __ballot_sync(0xffffffff, keep);
        if (lane == 0 && row < len) {
            words[row / 32] = bits;
        }
        count += keep;
    }
    const uint64_t prefix = filter_exclusive_sum(count, warp_sums);
    if (threadIdx.x == FILTER_THREADS - 1) {
        tile_counts[blockIdx.x] = prefix + count;
    }
}

// The host only uses this entry point when every comparison belongs to one field/type.
// Count the ballots already needed for the bitmap, rather than computing unused row prefixes.
template <typename T>
__device__ __forceinline__ void filter_typed_mask(const FilterPredicates &predicates,
                                                  uint32_t num_predicates,
                                                  uint64_t len,
                                                  uint32_t *words,
                                                  uint64_t *tile_counts) {
    __shared__ uint32_t warp_counts[FILTER_THREADS / 32];
    const auto *input = reinterpret_cast<const T *>(predicates.comparisons[0].input);
    const uint64_t tile_start = static_cast<uint64_t>(blockIdx.x) * FILTER_TILE_ROWS;
    const uint32_t lane = threadIdx.x % 32;
    uint32_t count = 0;
    for (uint32_t offset = threadIdx.x; offset < FILTER_TILE_ROWS; offset += FILTER_THREADS) {
        const uint64_t row = tile_start + offset;
        bool keep = row < len;
        if (keep) {
            const T value = input[row];
            for (uint32_t index = 0; index < num_predicates; ++index) {
                const FilterComparison &comparison = predicates.comparisons[index];
                keep = keep && filter_compare(value, filter_literal<T>(comparison.literal), comparison.op);
            }
        }
        const uint32_t bits = __ballot_sync(0xffffffff, keep);
        if (lane == 0) {
            if (row < len) {
                words[row / 32] = bits;
            }
            count += __popc(bits);
        }
    }
    if (lane == 0) {
        warp_counts[threadIdx.x / 32] = count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // A tile has at most 2048 rows; global counts and prefixes remain u64.
        uint32_t total = 0;
        for (uint32_t warp = 0; warp < FILTER_THREADS / 32; ++warp) {
            total += warp_counts[warp];
        }
        tile_counts[blockIdx.x] = total;
    }
}

#define GENERATE_TYPED_FILTER_MASK(suffix, Type)                                                             \
    extern "C" __global__ void exact_filter_mask_##suffix(FilterPredicates predicates,                       \
                                                          uint32_t num_predicates,                           \
                                                          uint64_t len,                                      \
                                                          uint32_t *__restrict words,                        \
                                                          uint64_t *__restrict tile_counts) {                \
        filter_typed_mask<Type>(predicates, num_predicates, len, words, tile_counts);                        \
    }

GENERATE_TYPED_FILTER_MASK(u8, uint8_t)
GENERATE_TYPED_FILTER_MASK(u16, uint16_t)
GENERATE_TYPED_FILTER_MASK(u32, uint32_t)
GENERATE_TYPED_FILTER_MASK(u64, uint64_t)
GENERATE_TYPED_FILTER_MASK(i8, int8_t)
GENERATE_TYPED_FILTER_MASK(i16, int16_t)
GENERATE_TYPED_FILTER_MASK(i32, int32_t)
GENERATE_TYPED_FILTER_MASK(i64, int64_t)
GENERATE_TYPED_FILTER_MASK(f32, float)
GENERATE_TYPED_FILTER_MASK(f64, double)
#undef GENERATE_TYPED_FILTER_MASK

// Scan groups of 256 tile counts. When there is more than one group, a second
// invocation scans the group totals. The scatter adds both levels of prefixes.
extern "C" __global__ void exact_filter_prefix(const uint64_t *__restrict counts,
                                                uint64_t len,
                                                uint64_t *__restrict offsets,
                                                uint64_t *__restrict group_counts) {
    __shared__ uint64_t warp_sums[FILTER_THREADS / 32];
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * FILTER_THREADS + threadIdx.x;
    const uint64_t count = index < len ? counts[index] : 0;
    const uint64_t prefix = filter_exclusive_sum(count, warp_sums);
    if (index < len) {
        offsets[index] = prefix;
    }
    if (threadIdx.x == FILTER_THREADS - 1) {
        group_counts[blockIdx.x] = prefix + count;
    }
}

template <typename T>
__device__ void filter_scatter_column(const FilterColumn &column,
                                      const uint16_t *rows,
                                      uint32_t count,
                                      uint64_t tile_start,
                                      uint64_t output_start) {
    const auto *input = reinterpret_cast<const T *>(column.input);
    auto *output = reinterpret_cast<T *>(column.output);
    for (uint32_t index = threadIdx.x; index < count; index += FILTER_THREADS) {
        output[output_start + index] = input[tile_start + rows[index]];
    }
}

// Construct stable tile-local selected row indices once, then reuse them across all
// columns. Writes are contiguous; reads stay within a 2048-row input tile. No global
// per-row rank/index array and no repeated per-column scan of the bitmap are needed.
extern "C" __global__ void exact_filter_scatter(FilterColumns columns,
                                                 uint32_t num_columns,
                                                 uint64_t len,
                                                 const uint32_t *__restrict words,
                                                 const uint64_t *__restrict tile_offsets,
                                                 const uint64_t *__restrict group_offsets) {
    __shared__ uint64_t warp_sums[FILTER_THREADS / 32];
    __shared__ uint16_t rows[FILTER_TILE_ROWS];
    __shared__ uint32_t selected;
    const uint64_t tile_start = static_cast<uint64_t>(blockIdx.x) * FILTER_TILE_ROWS;
    const uint64_t first_row = tile_start + static_cast<uint64_t>(threadIdx.x) * 32;
    uint32_t bits = threadIdx.x < FILTER_TILE_ROWS / 32 && first_row < len ? words[first_row / 32] : 0;
    const uint32_t count = __popc(bits);
    uint32_t rank = static_cast<uint32_t>(filter_exclusive_sum(count, warp_sums));
    if (threadIdx.x == FILTER_THREADS - 1) {
        selected = rank + count;
    }
    while (bits != 0) {
        const uint32_t bit = __ffs(bits) - 1;
        rows[rank++] = static_cast<uint16_t>(threadIdx.x * 32 + bit);
        bits &= bits - 1;
    }
    __syncthreads();
    if (selected == 0) {
        return;
    }
    uint64_t output_start = tile_offsets[blockIdx.x];
    if (gridDim.x > FILTER_THREADS) {
        output_start += group_offsets[blockIdx.x / FILTER_THREADS];
    }
    for (uint32_t column = 0; column < num_columns; ++column) {
        const FilterColumn &descriptor = columns.columns[column];
        switch (descriptor.width) {
        case 1: filter_scatter_column<uint8_t>(descriptor, rows, selected, tile_start, output_start); break;
        case 2: filter_scatter_column<uint16_t>(descriptor, rows, selected, tile_start, output_start); break;
        case 4: filter_scatter_column<uint32_t>(descriptor, rows, selected, tile_start, output_start); break;
        case 8: filter_scatter_column<uint64_t>(descriptor, rows, selected, tile_start, output_start); break;
        }
    }
}
