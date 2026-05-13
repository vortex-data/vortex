#pragma once
#include <onpair/core/types.h>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Packed token store.
//
// Compressed strings are stored as a continuous LSB-first bit-packed stream of
// token ids inside a vector<uint64_t>.  All tokens in the column share the same
// fixed bit-width (9–16 bits).
//
// Per-string boundaries are recorded as token-stream indices (not byte offsets),
// using the Arrow-style sentinel convention: boundaries has n+1 entries for n
// strings, where boundaries[i] is the token-stream start of string i and
// boundaries[n] is the total token count.
//
// Store is write-once during encoding (BitWriter fills it) and then
// consumed read-only via StoreView.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair {

// Owns the bit-packed token stream and per-string boundaries.
struct Store {
    BitWidth              bit_width;        // Immutable after first write (9–16)
    std::vector<uint64_t> packed;           // LSB-first bit-packed token stream
    std::vector<uint32_t> boundaries;       // boundaries[i] = token-index start of string i
                                            // boundaries.back() = total token count

    size_t num_strings()  const noexcept {
        return boundaries.empty() ? 0 : boundaries.size() - 1;
    }
    size_t num_tokens()   const noexcept {
        return boundaries.empty() ? 0 : boundaries.back();
    }
    size_t bytes_used()   const noexcept {
        if(boundaries.empty()) return 0;
        size_t total_bits = num_tokens() * bit_width;
        size_t packed_bytes = (total_bits + 7) / 8;
        return packed_bytes + boundaries.size() * sizeof(uint32_t);
    }
};

} // namespace onpair
