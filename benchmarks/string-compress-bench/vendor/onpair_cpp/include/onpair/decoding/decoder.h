#pragma once
#include <onpair/core/dictionary_view.h>
#include <onpair/core/store_view.h>
#include <onpair/decoding/token_cursor.h>
#include <onpair/decoding/detail/decode_all.h>
#include <cstring>

namespace onpair::decoding {

// ─── Decompression ───────────────────────────────────────────────────────────
// Free functions that decompress a compressed column.  The bit-packed token
// stream lives in StoreView; token byte sequences live in DictionaryView.
//
// Two modes:
//
//   decompress(sv, dv, idx, buf)   — random access; resolves bit width once,
//                                    then walks the token span for string `idx`.
//
//   decompress_all(sv, dv, buf)    — bulk; delegates to decode_all<Bits>,
//                                    a branch-free, maximally unrolled loop.
//
// Both modes copy exactly MAX_TOKEN_SIZE bytes per token (over-copy), so buf
// must have DECOMPRESS_BUFFER_PADDING bytes beyond the true string length.

// ── Random access ─────────────────────────────────────────────────────────────
// Decompresses string `idx` into `buf`.
// Returns the number of bytes written.
inline size_t decompress(StoreView sv, DictionaryView dv,
                         size_t idx, uint8_t* buf) noexcept
{
    auto span = sv.string_span(idx);
    const uint8_t*  bytes   = dv.raw_bytes();
    const uint32_t* offsets = dv.raw_offsets();
    size_t written = 0;
    dispatch_bits(sv.bits(), [&](auto bits) noexcept {
        TokenCursor<bits.value> cursor(
            sv.packed_data(), span);
        while (cursor.has_more()) {
            const Token    t   = cursor.next();
            const uint32_t off = offsets[t];
            std::memcpy(buf + written, bytes + off, MAX_TOKEN_SIZE);
            written += offsets[t + 1] - off;
        }
    });
    return written;
}

// ── Bulk decompression ────────────────────────────────────────────────────────
// Decompresses the entire column into `buf` sequentially.
//
// Returns total bytes written.
inline size_t decompress_all(StoreView sv, DictionaryView dv,
                             uint8_t* buf) noexcept
{
    const uint32_t total = static_cast<uint32_t>(sv.num_tokens());
    return dispatch_bits(sv.bits(), [&](auto bits) {
        return decode_all<bits.value>(sv.packed_data(),
                                     dv.raw_bytes(), dv.raw_offsets(),
                                     total, buf);
    });
}

// ── Bulk decompression with Arrow-style offsets ───────────────────────────────
// Decompresses the entire column into `buf` and fills `out_offsets` with
// Arrow-style byte offsets.
//
// Returns total bytes written.
inline size_t decompress_all(StoreView sv, DictionaryView dv,
                             uint8_t* buf, uint32_t* out_offsets) noexcept
{
    const uint32_t total = static_cast<uint32_t>(sv.num_tokens());
    const size_t   n     = sv.num_strings();
    return dispatch_bits(sv.bits(), [&](auto bits) {
        return decode_all<bits.value>(sv.packed_data(), sv.boundaries(),
                                     dv.raw_bytes(), dv.raw_offsets(),
                                     total, n, buf, out_offsets);
    });
}

} // namespace onpair::decoding
