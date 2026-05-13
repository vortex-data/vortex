#pragma once
#include <onpair/search/automata/token_automaton.h>
#include <onpair/search/automata/token_stream.h>
#include <onpair/decoding/token_cursor.h>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// drive — feed every token from a stream into an automaton
// ─────────────────────────────────────────────────────────────────────────────
// Resets the automaton, walks the stream, and returns the verdict.
// Early-exits via is_dead() when the automaton is DeadDetectable.

template<TokenAutomaton A, TokenStream S>
bool drive(A& aut, S& stream) {
    aut.reset();
    while (stream.has_more()) {
        aut.step(stream.next());
        if constexpr (DeadDetectable<A>) {
            if (aut.is_dead()) break;
        }
    }
    return aut.is_accepted();
}

// ─────────────────────────────────────────────────────────────────────────────
// scan_impl — column scan loop, monomorphised on Bits
// ─────────────────────────────────────────────────────────────────────────────
// Called from ColumnView::scan() after the bit-width switch resolves Bits to
// a compile-time constant.

namespace detail {

template<BitWidth Bits, TokenAutomaton A, std::invocable<size_t> F>
void scan_impl(A& aut, const uint64_t* ONPAIR_RESTRICT packed,
               const uint32_t* ONPAIR_RESTRICT bounds,
               size_t n, F&& on_match)
{
    decoding::TokenCursor<Bits> cursor(packed);
    for (size_t i = 0; i < n; ++i) {
        cursor.reset_to(StreamSpan{bounds[i], bounds[i + 1]});
        if (drive(aut, cursor)) on_match(i);
    }
}

} // namespace detail
} // namespace onpair::search
