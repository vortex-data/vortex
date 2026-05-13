#pragma once
#include <onpair/search/automata/token_automaton.h>
#include <onpair/core/types.h>
#include <onpair/core/dictionary_view.h>
#include <onpair/search/detail/tokenize.h>
#include <cstdint>
#include <string_view>
#include <vector>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// PrefixAutomaton
// ─────────────────────────────────────────────────────────────────────────────
// Token-level automaton for prefix search (SQL `WHERE col LIKE 'prefix%'`).
//
// Algorithm:
//   1. Tokenize the prefix and precompute valid-divergence intervals.
//   2. step(): compare each incoming token against the query sequence.
//      - Exact match at position i → advance.
//      - Mismatch at position i → check if token falls in the precomputed
//        interval [lb, ub] for that position (valid divergence → accept).
//      - All query tokens consumed → accept (remaining tokens are irrelevant).
//
// DeadDetectable: is_dead() returns true as soon as the automaton reaches a
// terminal state (accepted or rejected).  Once all query tokens are matched
// or a divergence decision is made, the result is final.

class PrefixAutomaton {
public:
    PrefixAutomaton(std::string_view prefix, DictionaryView dv);

    // ── TokenAutomaton / DeadDetectable interface ───────────────────────────
    void step(Token t) noexcept {
        if (is_dead()) return;

        if (t != query_tokens_[pos_]) {
            status_ = intervals_[pos_].contains(t) ? Status::accepted : Status::rejected;
            return;
        }

        if(++pos_ == query_tokens_.size())
            status_ = Status::accepted;
    }

    bool is_accepted() const noexcept { return status_ == Status::accepted; }

    void reset() noexcept {
        pos_ = 0;
        status_ = query_tokens_.empty() ? Status::accepted : Status::matching;
    }

    bool is_dead() const noexcept { return status_ != Status::matching; }

    // ── Accessors ───────────────────────────────────────────────────────────
    size_t query_length() const noexcept { return query_tokens_.size(); }

private:
    enum class Status : uint8_t { matching, accepted, rejected };

    std::vector<Token> query_tokens_;
    std::vector<TokenRange> intervals_;

    size_t pos_    = 0;
    Status status_ = Status::matching;
};

// ─── Implementation ─────────────────────────────────────────────────────────

inline PrefixAutomaton::PrefixAutomaton(std::string_view prefix,
                                         DictionaryView dv)
    : query_tokens_(detail::tokenize(prefix, dv))
{
    const size_t q_len = query_tokens_.size();
    intervals_.resize(q_len);

    if (q_len == 0) {
        status_ = Status::accepted;
        return;
    }

    const auto* pfx_data = reinterpret_cast<const uint8_t*>(prefix.data());

    size_t current_pos = 0;

    for (size_t i = 0; i < q_len; ++i) {
        const uint8_t* suffix_ptr = pfx_data + current_pos;
        const size_t   suffix_len = prefix.size() - current_pos;

        intervals_[i] = dv.prefix_range(suffix_ptr, suffix_len);
        current_pos += dv.token_size(query_tokens_[i]);
    }
}

} // namespace onpair::search
