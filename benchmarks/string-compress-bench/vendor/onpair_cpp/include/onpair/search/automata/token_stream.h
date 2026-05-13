#pragma once
#include <onpair/core/types.h>
#include <concepts>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// TokenStream concept
// ─────────────────────────────────────────────────────────────────────────────
// Any pull-model source of tokens.  TokenCursor<Bits> satisfies this, but so
// does any test double or alternative decoder.

template<typename S>
concept TokenStream = requires(S s) {
    { s.has_more() } -> std::convertible_to<bool>;
    { s.next()     } -> std::same_as<Token>;
};

} // namespace onpair::search
