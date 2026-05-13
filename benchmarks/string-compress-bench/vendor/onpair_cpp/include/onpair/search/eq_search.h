#pragma once
#include <onpair/core/types.h>
#include <onpair/core/dictionary_view.h>
#include <onpair/decoding/token_cursor.h>
#include <onpair/search/detail/tokenize.h>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// EQSearch
// ─────────────────────────────────────────────────────────────────────────────
// Finds all strings in a compressed column that are exactly equal to a given
// value (SQL `WHERE col = 'value'`).
//
// Algorithm:
//   1. Tokenize the query value by greedy longest-match against the sorted
//      dictionary.
//   2. For each string: fast rejection if the token count differs; then 
//      token-by-token equality check with early exit on the first mismatch.

class EQSearch {
public:
    // Constructs from a value and a sorted dictionary.
    // An empty value matches only empty strings (zero tokens).
    EQSearch(std::string_view value, DictionaryView dv)
        : query_tokens_(detail::tokenize(value, dv))
    {}

    // Number of query tokens.
    size_t query_length() const noexcept { return query_tokens_.size(); }

    // ── Scan interface ──────────────────────────────────────────────────────

    template<BitWidth Bits>
    bool matches(decoding::TokenCursor<Bits>& cursor) const noexcept;

    template<BitWidth Bits, std::invocable<size_t> F>
    void scan(const uint64_t* ONPAIR_RESTRICT packed,
              const uint32_t* ONPAIR_RESTRICT bounds,
              size_t n, F&& on_match) const;

private:
    std::vector<Token> query_tokens_;
};

// ─── Implementation ─────────────────────────────────────────────────────────

template<BitWidth Bits>
bool EQSearch::matches(decoding::TokenCursor<Bits>& cursor) const noexcept
{
    const uint32_t n_query = static_cast<uint32_t>(query_tokens_.size());

    // Fast reject: token counts must match.
    if (cursor.remaining() != n_query) return false;

    const Token* query = query_tokens_.data();

    for (uint32_t i = 0; i < n_query; ++i) {
        if (cursor.next() != query[i]) return false;
    }
    return true;
}

template<BitWidth Bits, std::invocable<size_t> F>
void EQSearch::scan(const uint64_t* ONPAIR_RESTRICT packed,
                            const uint32_t* ONPAIR_RESTRICT bounds,
                            size_t n, F&& on_match) const
{
    decoding::TokenCursor<Bits> cursor(packed);
    for (size_t i = 0; i < n; ++i) {
        cursor.reset_to(StreamSpan{bounds[i], bounds[i + 1]});
        if (matches<Bits>(cursor))
            on_match(i);
    }
}

} // namespace onpair::search
