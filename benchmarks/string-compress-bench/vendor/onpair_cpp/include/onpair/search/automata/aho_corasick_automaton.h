#pragma once
#include <onpair/search/aho_corasick_trie.h>
#include <onpair/search/automata/token_automaton.h>
#include <onpair/core/dictionary_view.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// AhoCorasickAutomaton  (eager)
// ─────────────────────────────────────────────────────────────────────────────
// Token-level Aho-Corasick automaton with fully eager precomputation.
// Answers: "does this string contain ANY of the given patterns?"
//
// Construction:
//   Base pass:   for each dictionary token t, walk the byte-level AC Trie
//                from ROOT_STATE through t's bytes.  Record exit state —
//                cost is proportional to total dictionary byte volume.
//   Sparse pass: for every AC state j ≠ ROOT_STATE, traverse the
//                dictionary's implicit trie tracking two AC evolutions in
//                parallel (from j and from ROOT_STATE), pruning when they
//                converge.  Store exceptions as sorted range vectors.
//                Cost is proportional to the number of AC states × relevant
//                dictionary branches.
//
// Both passes run at construction, so step() is a single table lookup with
// a short linear scan over sparse ranges — no trie walking at query time.
//
// Best when query volume is large relative to AC state count, so that the
// upfront sparse-pass cost is amortised.  For workloads where most AC
// states are never reached, see AhoCorasickLazyAutomaton.

class AhoCorasickAutomaton {
public:
    using State = AhoCorasickTrie::State;

    // Convenience constructor: Builds the Trie internally.
    AhoCorasickAutomaton(std::span<const std::string_view> patterns, DictionaryView dict)
        : AhoCorasickAutomaton(AhoCorasickTrie(patterns), dict) {}

    // High-performance constructor: Reuses an existing compiled Trie.
    AhoCorasickAutomaton(const AhoCorasickTrie& trie, DictionaryView dict);

    // ── TokenAutomaton / DeadDetectable interface ───────────────────────────
    void step(Token t) noexcept {
        if (hit_) return;

        if (state_ != ROOT_STATE) {
            const uint32_t start = sparse_offsets_[state_];
            const uint32_t end   = sparse_offsets_[state_ + 1];
            
            for (uint32_t i = start; i < end; ++i) {
                const auto& r = sparse_ranges_[i];
                if (t < r.begin) break;
                if (t <= r.last) {
                    State target_state = sparse_targets_[i];
                    hit_ = (target_state == HIT);
                    state_ = target_state;
                    return;
                }
            }
        }
        
        State target_state = base_[t];
        hit_ = (target_state == HIT);
        state_ = target_state;
    }

    bool is_accepted()          const noexcept { return hit_; }
    bool is_dead()              const noexcept { return hit_; }
    void reset()                      noexcept { state_ = ROOT_STATE; hit_ = all_match_; }

private:
    static constexpr State HIT = AhoCorasickTrie::NULL_STATE;
    static constexpr State ROOT_STATE = AhoCorasickTrie::ROOT_STATE;

    State state_ = ROOT_STATE;
    bool  hit_   = false;
    bool  all_match_ = false;

    // base_[token] = transition from AC ROOT_STATE.
    std::vector<State> base_;

    // Arrow-style SoA flattened sparse transitions grouped by AC state.
    std::vector<uint32_t>   sparse_offsets_;
    std::vector<TokenRange> sparse_ranges_;
    std::vector<State>      sparse_targets_;
};

// ─── Implementation ─────────────────────────────────────────────────────────

inline AhoCorasickAutomaton::AhoCorasickAutomaton(
    const AhoCorasickTrie& trie,
    DictionaryView dict)
{
    all_match_ = trie.is_accepting(ROOT_STATE);
    hit_       = all_match_;

    if(all_match_) return;

    const size_t num_states = trie.num_states();
    const size_t num_tokens = dict.num_tokens();

    // ── 1. Base pass: transitions from ROOT_STATE ─────────────────────────────
    base_.resize(num_tokens);
    for (Token t = 0; t < num_tokens; ++t) {
        const uint8_t* data = dict.data(t);
        const size_t   len  = dict.token_size(t);

        State s = ROOT_STATE;
        for (size_t i = 0; i < len; ++i) {
            s = trie.advance(s, data[i]);
            if (trie.is_accepting(s)) {
                s = HIT;
                break;
            }
        }

        base_[t] = s;
    }

    // ── 2. Sparse pass — dual-AC trie traversal ────────────────────────────
    //
    // For each AC states different from ROOT_STATE, traverse the dictionary's 
    // implicit trie tracking two AC states in parallel:
    //   state_j = state evolved from entry state j
    //   state_0 = state evolved from ROOT_STATE
    //
    // Pruning: when state_j == state_0, the subtree produces no exceptions.
    // Ranges are merged on-the-fly since tokens are visited in sorted order.

    sparse_offsets_.resize(num_states + 1, 0);
    size_t current_range_start = 0;

    // Extend last transition or push a new one.
    auto emit = [&](TokenRange range, State target_state) {
        if (sparse_ranges_.size() > current_range_start) {
            if (sparse_targets_.back() == target_state && sparse_ranges_.back().last + 1 == range.begin) {
                sparse_ranges_.back().last = range.last;
                return;
            }
        }
        sparse_ranges_.push_back(range);
        sparse_targets_.push_back(target_state);
    };

    // Evolve a state through one byte.
    auto evolve = [&](State state, uint8_t c) -> State {
        if (state == HIT) return HIT;
        State next = trie.advance(state, c);
        return trie.is_accepting(next) ? HIT : next;
    };

    auto traverse = [&](auto& self, TokenRange tr, size_t depth,
                        State state_j, State state_0) -> void {
        if (state_j == state_0 || tr.empty()) return;

        if (state_j == HIT) {
            Token i = tr.begin;
            while (i <= tr.last) {
                if (base_[i] != HIT) {
                    Token start = i;
                    while (i <= tr.last && base_[i] != HIT) ++i;
                    emit({start, static_cast<Token>(i - 1)}, HIT);
                } else {
                    ++i;
                }
            }
            return;
        }

        Token cur = tr.begin;
        while (cur <= tr.last && dict.token_size(cur) == depth) ++cur;
        
        if (cur > tr.begin) emit({tr.begin, static_cast<Token>(cur - 1)}, state_j);
        if (cur > tr.last)  return;

        while (cur <= tr.last) {
            uint8_t c = dict.data(cur)[depth];
            Token sub_hi = cur;
            while (sub_hi < tr.last && dict.data(static_cast<Token>(sub_hi + 1))[depth] == c) {
                ++sub_hi;
            }

            self(self, {cur, sub_hi}, depth + 1,
                 evolve(state_j, c), evolve(state_0, c));

            cur = static_cast<Token>(sub_hi + 1);
        }
    };

    std::vector<uint8_t> relevant_chars;
    sparse_offsets_[0] = 0;

    for (State j = 1; j < num_states; ++j) {
        current_range_start = sparse_ranges_.size();
        sparse_offsets_[j] = static_cast<uint32_t>(current_range_start);

        // Collect byte labels of trie children along the failure chain from j
        relevant_chars.clear();
        State u = j;
        while (u != ROOT_STATE) {
            for (uint8_t c : trie.edge_labels(u)) {
                relevant_chars.push_back(c);
            }
            u = trie.fail_link(u);
        }

        std::sort(relevant_chars.begin(), relevant_chars.end());
        relevant_chars.erase(std::unique(relevant_chars.begin(), relevant_chars.end()), relevant_chars.end());

        for (uint8_t byte : relevant_chars) {
            if (trie.advance(j, byte) == trie.advance(ROOT_STATE, byte)) {
                continue;
            }
            TokenRange range = dict.prefix_range(&byte, 1);
            if (range.empty()) continue;

            traverse(traverse, range, 1, 
                     evolve(j, byte), 
                     evolve(ROOT_STATE, byte));
        }
    }

    sparse_offsets_[num_states] = static_cast<uint32_t>(sparse_ranges_.size());
}

} // namespace onpair::search
