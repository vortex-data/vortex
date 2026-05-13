#pragma once
#include <onpair/search/automata/token_automaton.h>
#include <onpair/core/dictionary_view.h>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>
#include <algorithm>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// KmpAutomaton
// ─────────────────────────────────────────────────────────────────────────────
// Token-level KMP automaton for substring search (SQL LIKE '%pattern%').
//
// Construction:
//   1. Build byte-level KMP failure table from the pattern.
//   2. Base pass:   for each token t, run KMP from state 0 through t's bytes
//                   → base_[t].  Stored as a dense vector, with one State per 
//                   token.
//   3. Sparse pass: for each non-zero entry state j, find the tokens whose
//                   exit state differs from base_[t] and record them as sparse
//                   exception ranges.   Stored as flattened vector of (range, 
//                   target) pairs.
//
// Precondition: the pattern must be at most 255 bytes long, since KMP states
// (0 … pattern_length) are stored as uint8_t.  Exceeding this limit causes
// silent wraparound and undefined behaviour.

class KmpAutomaton {
public:
    using State = uint8_t;

    KmpAutomaton(std::string_view pattern, DictionaryView dict);

    // ── TokenAutomaton / DeadDetectable interface ───────────────────────────
    void step(Token t) noexcept {
        if (is_dead()) return;

        if (state_ > 0) {
            const auto* r   = sparse_.data() + offsets_[state_];
            const auto* end = sparse_.data() + offsets_[state_ + 1];
            for (; r != end; ++r) {
                if (t < r->range.begin) break;
                if (t <= r->range.last) { state_ = r->target; return; }
            }
        }
        state_ = base_[t];
    }

    bool is_accepted() const noexcept { return state_ == match_state_; }
    void reset()       noexcept       { state_ = 0; }
    bool is_dead()     const noexcept { return state_ == match_state_; }

    // ── Accessors (testing / introspection) ─────────────────────────────────
    size_t pattern_length()     const noexcept { return match_state_; }
    size_t sparse_range_count() const noexcept { return sparse_.size(); }

private:
    // Sparse transition: tokens in [range.begin, range.last] map to `target`.
    struct SparseTransition {
        TokenRange range;
        State      target;
    };

    State match_state_;
    State state_ = 0;

    // base_[token] = KMP exit state after consuming token's bytes from state 0.
    std::vector<State> base_;

    // Flattened sparse transitions grouped by entry state.
    // Transitions for state s live at sparse_[offsets_[s] .. offsets_[s+1]).
    std::vector<SparseTransition> sparse_;
    std::vector<uint16_t>         offsets_;  // size = match_state_ + 1
};

// ─── Implementation ─────────────────────────────────────────────────────────

inline KmpAutomaton::KmpAutomaton(std::string_view pattern, DictionaryView dict)
    : match_state_(static_cast<State>(pattern.size()))
{
    const size_t m = pattern.size();
    const size_t num_tokens = dict.num_tokens();

    if (m == 0) {
        base_.assign(num_tokens, 0);
        offsets_.assign(2, 0);
        return;
    }

    const auto* p = reinterpret_cast<const uint8_t*>(pattern.data());

    // ── 1. KMP failure table ────────────────────────────────────────────────
    std::vector<State> fail(m, 0);
    for (State i = 1, len = 0; i < m; ) {
        if (p[i] == p[len]) {
            fail[i++] = ++len;
        } else if (len > 0) {
            len = fail[len - 1];
        } else {
            fail[i++] = 0;
        }
    }

    // KMP transition: consume `len` bytes from state `s` (absorbing at m).
    auto step_bytes = [&](State s, const uint8_t* data, size_t len) -> State {
        for (size_t i = 0; i < len; ++i) {
            if (s == m) return static_cast<State>(m);
            while (s > 0 && p[s] != data[i]) s = fail[s - 1];
            if (p[s] == data[i]) ++s;
        }
        return s;
    };

    // ── 2. Base pass ────────────────────────────────────────────────────────
    base_.resize(num_tokens);
    {
        const uint8_t*  bytes   = dict.raw_bytes();
        const uint32_t* offsets = dict.raw_offsets();
        const uint8_t   p0     = p[0];
        for (size_t t = 0; t < num_tokens; ++t) {
            const uint8_t* tok     = bytes + offsets[t];
            const size_t   tok_len = offsets[t + 1] - offsets[t];
            if (!std::memchr(tok, p0, tok_len)) {
                base_[t] = 0;
                continue;
            }
            base_[t] = step_bytes(0, tok, tok_len);
        }
    }

    // ── 3. Sparse pass — dual-KMP trie traversal ───────────────────────────
    //
    // For each entry state j > 0, we traverse the implicit trie of the sorted
    // dictionary, tracking two KMP states in parallel:
    //   kmp_j = state evolved from entry state j
    //   kmp_0 = state evolved from state 0
    //
    // Pruning: when kmp_j == kmp_0, the subtree produces no sparse entries.
    // Ranges are merged on-the-fly since tokens are visited in ascending order.

    offsets_.resize(m + 1, 0);
    size_t range_start = 0;

    // Extend last transition or push a new one.
    auto emit = [&](TokenRange range, State target) {
        if (sparse_.size() > range_start) {
            auto& last = sparse_.back();
            if (last.target == target && last.range.last + 1 == range.begin) {
                last.range.last = range.last;
                return;
            }
        }
        sparse_.push_back({range, target});
    };

    auto traverse = [&](auto& self, TokenRange tr,
                        size_t depth, State kmp_j, State kmp_0) -> void {
        if (kmp_j == kmp_0 || tr.empty()) return;

        // Full match: override tokens whose base differs from m.
        if (kmp_j == m) {
            const auto exit = static_cast<State>(m);
            Token i = tr.begin;
            while (i <= tr.last) {
                if (base_[i] != exit) {
                    Token start = i;
                    while (i <= tr.last && base_[i] != exit) ++i;
                    emit({start, static_cast<Token>(i - 1)}, exit);
                } else {
                    ++i;
                }
            }
            return;
        }

        // Leaf tokens (length == depth) all share exit state kmp_j.
        Token cur = tr.begin;
        while (cur <= tr.last && dict.token_size(cur) == depth)
            ++cur;
        if (cur > tr.begin)
            emit({tr.begin, static_cast<Token>(cur - 1)}, kmp_j);
        if (cur > tr.last) return;

        // Recurse into subtrees partitioned by byte at `depth`.
        while (cur <= tr.last) {
            uint8_t c = dict.data(cur)[depth];
            Token sub_hi = cur;
            while (sub_hi < tr.last &&
                   dict.data(static_cast<Token>(sub_hi + 1))[depth] == c)
                ++sub_hi;

            self(self, {cur, sub_hi}, depth + 1,
                 step_bytes(kmp_j, &c, 1), step_bytes(kmp_0, &c, 1));

            cur = static_cast<Token>(sub_hi + 1);
        }
    };

    std::vector<uint8_t> relevant_chars;
    relevant_chars.reserve(m);

    for (State j = 1; j < m; ++j) {
        range_start = sparse_.size();
        offsets_[j] = static_cast<uint16_t>(range_start);

        // Collect relevant first bytes from the failure chain for each state j.
        // Only bytes p[s] along the chain j → fail[j-1] → ... → 0 can cause a
        // different KMP transition from state j vs state 0, so we skip all others.
        relevant_chars.clear();
        {
            State s = j;
            while (s > 0) {
                relevant_chars.push_back(p[s]);
                s = fail[s - 1];
            }
        }

        // Deduplicate.
        std::sort(relevant_chars.begin(), relevant_chars.end());
        relevant_chars.erase(
            std::unique(relevant_chars.begin(), relevant_chars.end()),
            relevant_chars.end());

        for (uint8_t byte : relevant_chars) {
            TokenRange range = dict.prefix_range(&byte, 1);
            if (range.empty()) continue;

            traverse(traverse, range, 1,
                     step_bytes(j, &byte, 1),
                     step_bytes(0, &byte, 1));
        }
    }

    offsets_[m] = static_cast<uint16_t>(sparse_.size());
}

} // namespace onpair::search
