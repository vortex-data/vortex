#pragma once
#include <onpair/core/types.h>
#include <vector>
#include <cstdint>
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// Dictionary storage.
//
// The dictionary maps each Token (uint16_t id) to its byte sequence.
// Memory layout: a flat `bytes` buffer + an `offsets` index, identical to the
// Arrow binary layout.  Token i occupies bytes[offsets[i]..offsets[i+1]).
//
// Tokens are always stored in lexicographic order of their byte sequences.
// Sorting is performed once at compression time and enables optimized query
// operations that exploit the ordering of token IDs (e.g. prefix range
// lookups).
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair {

struct Dictionary {
    // Flat concatenation of all token byte sequences.
    //
    // ── Decoder padding ─────────────────────────────────────────────────────
    // The decoder uses a fixed-size memcpy of MAX_TOKEN_SIZE bytes per token
    // regardless of the token's true length (over-copy optimisation).  For the
    // last token in the buffer (at offset offsets.back() - last_len), this
    // over-copy extends (MAX_TOKEN_SIZE - last_len) bytes past the true data.
    //
    // To keep every over-copy within the allocated buffer, call
    // pad_for_decoder() once — after all token bytes have been inserted.
    // It appends exactly (MAX_TOKEN_SIZE - last_token_len) zero bytes.
    //
    // Canonical sizes:
    //   • offsets.back()  — logical byte count (true token data, no padding)
    //   • bytes.size()    — allocated byte count (may include padding)
    //
    // bytes_used() and write_to() always use offsets.back() so they remain
    // correct regardless of whether padding has been applied.
    std::vector<uint8_t>  bytes;

    // offsets[i]..offsets[i+1] = byte range of token i in `bytes`.
    // Invariant: offsets[0] == 0, offsets.size() == num_tokens + 1.
    std::vector<uint32_t> offsets;

    size_t num_tokens() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }

    // Returns the logical dictionary size (true token bytes + offsets array).
    // Uses offsets.back() — not bytes.size() — so it is unaffected by padding.
    size_t bytes_used() const noexcept {
        const size_t true_bytes = offsets.empty() ? 0 : offsets.back();
        // Dictionaries with ≤2^12 tokens span at most 256×1 + 3840×16 = 61696
        // bytes (256 single-byte tokens are always present), so u16 offsets
        // suffice; larger dictionaries require u32.
        // TODO: offsets is always u32 — make it a variant to exploit this.
        const size_t offsets_bytes = offsets.size() * sizeof(uint32_t);
        return true_bytes + offsets_bytes;
    }

    // Append (MAX_TOKEN_SIZE - last_token_len) zero bytes so the decoder can
    // safely over-copy MAX_TOKEN_SIZE bytes starting from any token offset.
    // Idempotent: if bytes.size() > offsets.back() the padding is already in
    // place and the call is a no-op.
    // No-op when the last token is exactly MAX_TOKEN_SIZE bytes long.
    void pad_for_decoder() {
        if (offsets.size() < 2) return;
        if (bytes.size() > offsets.back()) return;  // already padded
        const size_t last_len = offsets.back() - offsets[offsets.size() - 2];
        bytes.resize(bytes.size() + (MAX_TOKEN_SIZE - last_len), 0);
    }
};

} // namespace onpair
