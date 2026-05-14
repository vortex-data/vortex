// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// C ABI over the OnPair C++ library. All functions are nothrow; failures are
// signalled by a non-zero return code, with the caller responsible for any
// out-parameter allocations.

#ifndef VORTEX_ONPAIR_SHIM_H
#define VORTEX_ONPAIR_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OnPairColumnHandle OnPairColumnHandle;

typedef enum OnPairStatus {
    ONPAIR_OK = 0,
    ONPAIR_ERR_INVALID_ARG = 1,
    ONPAIR_ERR_BAD_FORMAT = 2,
    ONPAIR_ERR_OUT_OF_RANGE = 3,
    ONPAIR_ERR_OOM = 4,
    ONPAIR_ERR_INTERNAL = 99,
} OnPairStatus;

// Training configuration. `bits` must be in [9, 16]; `dict_12` corresponds to
// bits = 12. `threshold` is the dynamic frequency threshold (smaller values
// produce larger dictionaries).
typedef struct OnPairTrainingConfig {
    uint32_t bits;
    double   threshold;
    uint64_t seed;
} OnPairTrainingConfig;

// `bytes` is the concatenation of all input strings; `offsets` has length `n + 1`
// such that the i-th string spans `bytes[offsets[i] .. offsets[i + 1]]`.
//
// On success, *out_handle is set to an owning handle that must be released with
// onpair_column_free.
OnPairStatus onpair_column_compress(
    const uint8_t* bytes,
    const uint64_t* offsets,
    size_t n,
    OnPairTrainingConfig config,
    OnPairColumnHandle** out_handle);

// Deserialize a previously-serialized OnPair column. `data` must contain the
// magic header `ONPAIR01` produced by onpair_column_serialize.
OnPairStatus onpair_column_deserialize(
    const uint8_t* data,
    size_t len,
    OnPairColumnHandle** out_handle);

// Serialize an OnPair column to a byte vector. The caller must free the
// returned buffer with onpair_buffer_free.
OnPairStatus onpair_column_serialize(
    const OnPairColumnHandle* handle,
    uint8_t** out_data,
    size_t* out_len);

void onpair_column_free(OnPairColumnHandle* handle);
void onpair_buffer_free(uint8_t* data, size_t len);

// Number of rows in the compressed column.
size_t onpair_column_len(const OnPairColumnHandle* handle);
// Bits-per-token the column was compressed with (9..=16).
uint32_t onpair_column_bits(const OnPairColumnHandle* handle);
// Dictionary size in entries.
size_t onpair_column_dict_size(const OnPairColumnHandle* handle);

// Decompress the row at `row_id` into `out_buf`. `out_buf` must have at least
// `out_capacity` bytes. On success `*out_len` holds the number of bytes
// written. Returns ONPAIR_ERR_OUT_OF_RANGE if `row_id` is out of bounds or
// ONPAIR_ERR_OOM if `out_capacity` is too small.
OnPairStatus onpair_column_decompress(
    const OnPairColumnHandle* handle,
    size_t row_id,
    uint8_t* out_buf,
    size_t out_capacity,
    size_t* out_len);

// Upper bound on the size of any single decompressed row, including the
// over-copy padding the C++ decoder requires.
size_t onpair_column_decompress_capacity(const OnPairColumnHandle* handle);

// --- Compressed-domain predicate pushdown ---------------------------------
//
// All `*_into` predicates write a bitmap of length `n` into `out_bits`
// (one bit per row, LSB-first, packed into bytes; the caller must provide
// at least `(n + 7) / 8` bytes).

OnPairStatus onpair_column_equals_into(
    const OnPairColumnHandle* handle,
    const uint8_t* needle,
    size_t needle_len,
    uint8_t* out_bits);

OnPairStatus onpair_column_starts_with_into(
    const OnPairColumnHandle* handle,
    const uint8_t* needle,
    size_t needle_len,
    uint8_t* out_bits);

OnPairStatus onpair_column_contains_into(
    const OnPairColumnHandle* handle,
    const uint8_t* needle,
    size_t needle_len,
    uint8_t* out_bits);

// --- Bulk dictionary access (for canonicalisation) ------------------------
//
// Copies the column's dictionary into the caller-provided buffer. The
// dictionary is laid out as a packed byte vector with parallel offsets
// (length `dict_size + 1`).
OnPairStatus onpair_column_dict_copy(
    const OnPairColumnHandle* handle,
    uint8_t* out_bytes,
    size_t bytes_capacity,
    uint64_t* out_offsets);

// Bytes occupied by the dictionary (sum of entry lengths).
size_t onpair_column_dict_bytes(const OnPairColumnHandle* handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_ONPAIR_SHIM_H
