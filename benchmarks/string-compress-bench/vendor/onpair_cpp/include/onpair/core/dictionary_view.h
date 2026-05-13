#pragma once
#include <onpair/core/dictionary.h>
#include <onpair/core/types.h>
#include <optional>
#include <utility>
#include <cstring>

namespace onpair {

// ─────────────────────────────────────────────────────────────────────────────
// DictionaryView — non-owning, read-only view over a Dictionary.
//
// Provides O(1) random access to token byte sequences by id, and O(log n)
// prefix range lookups over the sorted dictionary via binary search.
//
// Passed by value to decoders, search automata, and tokenisers — no
// allocation, no ownership transfer.  Lifetime is tied to the underlying
// Dictionary.
// ─────────────────────────────────────────────────────────────────────────────

class DictionaryView {
public:
    /* implicit */ DictionaryView(const Dictionary& d) noexcept
        : dict_(d) {}

    ByteSpan span(Token id) const noexcept {
        return { dict_.offsets[id], dict_.offsets[id + 1] };
    }
    const uint8_t* data(Token id) const noexcept {
        return dict_.bytes.data() + dict_.offsets[id];
    }
    size_t token_size(Token id) const noexcept {
        return dict_.offsets[id + 1] - dict_.offsets[id];
    }
    size_t num_tokens() const noexcept { return dict_.num_tokens(); }

    // Raw pointers for decode loops operating directly on the arrays
    const uint8_t*  raw_bytes()   const noexcept { return dict_.bytes.data(); }
    const uint32_t* raw_offsets() const noexcept { return dict_.offsets.data(); }

    size_t bytes_used() const noexcept {
        return dict_.bytes_used();
    }

    // Returns the [lo, hi] token-id range whose byte sequences share `prefix`.
    TokenRange prefix_range(
        const uint8_t* prefix, size_t prefix_len) const noexcept;

private:
    const Dictionary& dict_;
};

} // namespace onpair
