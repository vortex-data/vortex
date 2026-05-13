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
// AhoCorasickLazyAutomaton
// ─────────────────────────────────────────────────────────────────────────────
// Token-level Aho-Corasick automaton with deferred sparse-state expansion.
// Answers: "does this string contain ANY of the given patterns?"
//
// Hybrid between AhoCorasickOnlineAutomaton (zero precomputation) and the
// fully eager AhoCorasickAutomaton (all states precomputed upfront).
//
// Construction:
//   Base pass:   for each dictionary token t, walk the byte-level AC Trie
//                from ROOT_STATE through t's bytes.  Record exit state —
//                cost is proportional to total dictionary byte volume.
//   Sparse pass: deferred.  State-specific transition ranges are computed
//                on first access and cached for subsequent visits.
//
// Best when the AC Trie has many states (many / long patterns) but only a
// subset is actually reached at query time — e.g. most patterns do not
// match.  The eager variant pays for every state upfront; the lazy variant
// amortises that cost across actual query traffic.

class AhoCorasickLazyAutomaton {
public:
    using State = AhoCorasickTrie::State;

    // Convenience constructor: Builds the Trie internally.
    AhoCorasickLazyAutomaton(std::span<const std::string_view> patterns, DictionaryView dict)
        : AhoCorasickLazyAutomaton(std::make_shared<AhoCorasickTrie>(patterns), dict) {}

    // High-performance constructor: Reuses an existing compiled Trie.
    AhoCorasickLazyAutomaton(std::shared_ptr<const AhoCorasickTrie> trie, DictionaryView dict);

    // ── TokenAutomaton / DeadDetectable interface ───────────────────────────
    void step(Token t) noexcept {
        if (hit_) return;

        if (state_ != ROOT_STATE) {
            if (sparse_remap_[state_] == UNEXPANDED) expand_state(state_);

            auto remapped = sparse_remap_[state_];
            const uint32_t start = sparse_offsets_[remapped];
            const uint32_t end   = sparse_offsets_[remapped + 1];

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

    bool   is_accepted()        const noexcept { return hit_; }
    bool   is_dead()            const noexcept { return hit_; }
    void   reset()                    noexcept { state_ = ROOT_STATE; hit_ = all_match_; }

private:
    static constexpr State UNEXPANDED = AhoCorasickTrie::NULL_STATE;
    static constexpr State HIT = AhoCorasickTrie::NULL_STATE;
    static constexpr State ROOT_STATE = AhoCorasickTrie::ROOT_STATE;

    State state_ = ROOT_STATE;
    bool hit_   = false;
    bool all_match_ = false;

    void expand_state(State state);

    std::shared_ptr<const AhoCorasickTrie> trie_;
    DictionaryView                         dict_;

    // base_[token] = transition from AC ROOT_STATE.
    std::vector<State>            base_;

    // Arrow-style SoA flattened sparse transitions grouped by AC state.
    std::vector<State>      sparse_remap_;
    std::vector<uint32_t>   sparse_offsets_;
    std::vector<TokenRange> sparse_ranges_;
    std::vector<State>      sparse_targets_;
};

// ─── Implementation ─────────────────────────────────────────────────────────

inline AhoCorasickLazyAutomaton::AhoCorasickLazyAutomaton(
    std::shared_ptr<const AhoCorasickTrie> trie,
    DictionaryView dict)
    : trie_(std::move(trie)), dict_(dict)
{
    all_match_ = trie_->is_accepting(ROOT_STATE);
    hit_ = all_match_;

    if(all_match_) return;

    const size_t num_states = trie_->num_states();
    const size_t num_tokens = dict_.num_tokens();

    // ── 1. Base pass: transitions from ROOT_STATE ─────────────────────────────
    base_.resize(num_tokens);
    for (Token t = 0; t < num_tokens; ++t) {
        const uint8_t* data = dict_.data(t);
        const size_t   len  = dict_.token_size(t);

        State s = ROOT_STATE;
        for (size_t i = 0; i < len; ++i) {
            s = trie_->advance(s, data[i]);
            if (trie_->is_accepting(s)) {
                s = HIT;
                break;
            }
        }

        base_[t] = s;
    }

    // ── 2. (Lazy) Sparse pass  ────────────────────────────
    sparse_remap_.resize(num_states, UNEXPANDED);
    sparse_offsets_.push_back(0);
}

inline void AhoCorasickLazyAutomaton::expand_state(State state) {
    const State remapped = static_cast<State>(sparse_offsets_.size() - 1);
    sparse_remap_[state] = remapped;

    const uint32_t current_range_start = static_cast<uint32_t>(sparse_ranges_.size());

    // Extend last transition or push a new one.
    auto emit = [&](TokenRange range, State target_state) {
        if (sparse_ranges_.size() > current_range_start) {
            if (sparse_targets_.back() == target_state
                && sparse_ranges_.back().last + 1 == range.begin) {
                sparse_ranges_.back().last = range.last;
                return;
            }
        }
        sparse_ranges_.push_back(range);
        sparse_targets_.push_back(target_state);
    };

    // Evolve a state through one byte.
    auto evolve = [&](State s, uint8_t c) -> State {
        if (s == HIT) return HIT;
        State next = trie_->advance(s, c);
        return trie_->is_accepting(next) ? HIT : next;
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
        while (cur <= tr.last && dict_.token_size(cur) == depth) ++cur;

        if (cur > tr.begin) emit({tr.begin, static_cast<Token>(cur - 1)}, state_j);
        if (cur > tr.last)  return;

        while (cur <= tr.last) {
            uint8_t c = dict_.data(cur)[depth];
            Token sub_hi = cur;
            while (sub_hi < tr.last
                   && dict_.data(static_cast<Token>(sub_hi + 1))[depth] == c) {
                ++sub_hi;
            }

            self(self, {cur, sub_hi}, depth + 1,
                 evolve(state_j, c), evolve(state_0, c));

            cur = static_cast<Token>(sub_hi + 1);
        }
    };

    // Collect byte labels along the failure chain from state.
    std::vector<uint8_t> relevant_chars;
    State u = state;
    while (u != ROOT_STATE) {
        for (uint8_t c : trie_->edge_labels(u))
            relevant_chars.push_back(c);
        u = trie_->fail_link(u);
    }

    std::sort(relevant_chars.begin(), relevant_chars.end());
    relevant_chars.erase(std::unique(relevant_chars.begin(), relevant_chars.end()),
                         relevant_chars.end());

    for (uint8_t byte : relevant_chars) {
        if (trie_->advance(state, byte) == trie_->advance(ROOT_STATE, byte))
            continue;

        TokenRange range = dict_.prefix_range(&byte, 1);
        if (range.empty()) continue;

        traverse(traverse, range, 1,
                 evolve(state, byte),
                 evolve(ROOT_STATE, byte));
    }

    // Close this state's offset range.
    sparse_offsets_.push_back(static_cast<uint32_t>(sparse_ranges_.size()));
}

} // namespace onpair::search
