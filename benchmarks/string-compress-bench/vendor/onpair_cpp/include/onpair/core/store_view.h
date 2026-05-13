#pragma once
#include <onpair/core/store.h>

// ─────────────────────────────────────────────────────────────────────────────
// StoreView — non-owning, read-only view over Store.
//
// Passed by value to any consumer of the packed token stream — no allocation,
// no ownership transfer.  Lifetime is tied to the underlying Store.
//
// string_span(i) returns the [begin, end) token-stream range for string i.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair {

class StoreView {
public:
    /* implicit */ StoreView(const Store& s) noexcept : store_(s) {}

    BitWidth bits()       const noexcept { return store_.bit_width; }  // 9–16
    size_t num_strings()  const noexcept { return store_.num_strings(); }
    size_t num_tokens()   const noexcept { return store_.num_tokens(); }
    size_t bytes_used()   const noexcept { return store_.bytes_used(); }

    // Token-stream index range [begin, end) for string at position idx.
    // Precondition: idx < num_strings().
    StreamSpan string_span(size_t idx) const noexcept {
        return { store_.boundaries[idx], store_.boundaries[idx + 1] };
    }

    // Raw pointers for decode loops — avoids repeated vector.data() calls.
    const uint64_t* packed_data() const noexcept { return store_.packed.data(); }
    const uint32_t* boundaries()  const noexcept { return store_.boundaries.data(); }

private:
    const Store& store_;
};

} // namespace onpair
