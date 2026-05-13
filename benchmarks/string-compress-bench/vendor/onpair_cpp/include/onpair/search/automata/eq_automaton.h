#pragma once
#include <onpair/search/automata/token_automaton.h>
#include <onpair/core/dictionary_view.h>
#include <onpair/search/detail/tokenize.h>
#include <cstdint>
#include <string_view>
#include <vector>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// EqAutomaton
// ─────────────────────────────────────────────────────────────────────────────
// Token-level automaton for exact equality (SQL `WHERE col = 'value'`).
//
// Algorithm:
//   1. Tokenize the query value against the sorted dictionary.
//   2. step(): compare each incoming token against the query sequence.
//      Any mismatch or length difference → reject.
//   3. is_accepted(): true iff all query tokens matched and no extra tokens
//      were received.
//
// DeadDetectable: is_dead() returns true as soon as a mismatch occurs or the
// string has more tokens than the query.  The result is final at that point.

class EqAutomaton {
public:
    EqAutomaton(std::string_view value, DictionaryView dv)
        : query_tokens_(detail::tokenize(value, dv))
    {}

    // ── TokenAutomaton / DeadDetectable interface ───────────────────────────
    void step(Token t) noexcept {
        failed_ |= (pos_ >= query_tokens_.size()) || (t != query_tokens_[pos_]);
        ++pos_;
    }

    bool is_accepted() const noexcept {
        return !failed_ && pos_ == query_tokens_.size();
    }

    void reset() noexcept {
        pos_ = 0;
        failed_ = false;
    }

    bool is_dead() const noexcept { return failed_; }

    // ── Accessors ───────────────────────────────────────────────────────────
    size_t query_length() const noexcept { return query_tokens_.size(); }

private:
    std::vector<Token> query_tokens_;
    size_t pos_    = 0;
    bool   failed_ = false;
};

} // namespace onpair::search
