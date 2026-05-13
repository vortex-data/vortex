// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Thin C wrapper around `onpair_cpp` (vendor/onpair_cpp). Exposes a flat,
// stable C ABI consumed by `src/backends/onpair_cpp_ffi.rs`.

#include <onpair/api.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string_view>
#include <vector>

extern "C" {

struct OnPairHandle {
    onpair::OnPairColumn col;
};

struct OnPairIndexVec {
    size_t* data;
    size_t  len;
};

OnPairHandle* onpair_compress(
    const uint8_t* data,
    const uint32_t* offsets,
    size_t n_strings,
    uint8_t bits,
    uint32_t seed)
{
    onpair::encoding::TrainingConfig cfg;
    cfg.bits = static_cast<onpair::BitWidth>(bits);
    cfg.seed = seed;
    // Use the default dynamic threshold; it adapts to the corpus.
    auto col = onpair::OnPairColumn::compress(
        reinterpret_cast<const char*>(data), offsets, n_strings, cfg);
    auto* handle = new (std::nothrow) OnPairHandle{ std::move(col) };
    return handle;
}

void onpair_destroy(OnPairHandle* h) noexcept {
    delete h;
}

size_t onpair_bytes_used(const OnPairHandle* h) noexcept {
    return h->col.bytes_used();
}

size_t onpair_num_strings(const OnPairHandle* h) noexcept {
    return h->col.num_strings();
}

// Decompresses every string into `out_data`, writes a `n+1`-entry offset
// array into `out_offsets`, and returns the total byte count.
// Caller must size `out_data` to at least `bytes_used + DECOMPRESS_BUFFER_PADDING`
// and `out_offsets` to `n+1` slots.
size_t onpair_decompress_all(
    const OnPairHandle* h,
    uint8_t* out_data,
    uint32_t* out_offsets) noexcept
{
    auto v = h->col.view();
    return v.decompress_all(
        reinterpret_cast<char*>(out_data), out_offsets);
}

// Equality / contains / starts_with all return a heap-allocated index array.
// Free with `onpair_free_indices`.

static OnPairIndexVec take_vec(std::vector<size_t>&& v) {
    OnPairIndexVec out{};
    out.len = v.size();
    out.data = static_cast<size_t*>(std::malloc(out.len * sizeof(size_t)));
    if (out.data && out.len > 0) {
        std::memcpy(out.data, v.data(), out.len * sizeof(size_t));
    }
    return out;
}

OnPairIndexVec onpair_equals(
    const OnPairHandle* h,
    const char* needle,
    size_t needle_len) noexcept
{
    auto v = h->col.view();
    auto hits = v.equals(std::string_view(needle, needle_len));
    return take_vec(std::move(hits));
}

OnPairIndexVec onpair_contains(
    const OnPairHandle* h,
    const char* needle,
    size_t needle_len) noexcept
{
    auto v = h->col.view();
    auto hits = v.contains(std::string_view(needle, needle_len));
    return take_vec(std::move(hits));
}

OnPairIndexVec onpair_starts_with(
    const OnPairHandle* h,
    const char* prefix,
    size_t prefix_len) noexcept
{
    auto v = h->col.view();
    auto hits = v.starts_with(std::string_view(prefix, prefix_len));
    return take_vec(std::move(hits));
}

void onpair_free_indices(OnPairIndexVec v) noexcept {
    std::free(v.data);
}

// Padding the decompress buffer needs over the longest expected output. This
// must be queried from C++ because it is a `constexpr` derived from the
// max-token-size constant.
size_t onpair_decompress_padding() noexcept {
    return onpair::DECOMPRESS_BUFFER_PADDING;
}

} // extern "C"
