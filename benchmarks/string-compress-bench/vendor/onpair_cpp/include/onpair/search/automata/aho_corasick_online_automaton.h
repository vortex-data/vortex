#pragma once
#include <onpair/search/aho_corasick_trie.h>
#include <onpair/search/automata/token_automaton.h>
#include <onpair/core/dictionary_view.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// AhoCorasickOnlineAutomaton
// ─────────────────────────────────────────────────────────────────────────────
// Token-level Aho-Corasick automaton with zero token-level precomputation.
// Answers: "does this string contain ANY of the given patterns?"
//
// Construction:
//   No base or sparse pass.  The only upfront cost is building (or receiving)
//   the byte-level AhoCorasickTrie itself.
//
// At query time, step(Token t) walks the byte-level AC Trie through every
// byte of t, paying O(token_len) per token.  No per-state or per-token
// tables are materialised.
//
// Best for small row groups or narrow dictionaries where the overhead of
// precomputing token-level transitions (base + sparse passes) would exceed
// the total query cost.  Simplest of the three variants.

class AhoCorasickOnlineAutomaton {
public:
    using State = AhoCorasickTrie::State;

    // Convenience constructor: Builds the Trie internally.
    AhoCorasickOnlineAutomaton(std::span<const std::string_view> patterns, DictionaryView dict)
        : AhoCorasickOnlineAutomaton(std::make_shared<AhoCorasickTrie>(patterns), dict) {}

    // High-performance constructor: Reuses an existing compiled Trie.
    AhoCorasickOnlineAutomaton(std::shared_ptr<const AhoCorasickTrie> trie, DictionaryView dict)
        : trie_(std::move(trie))
        , dict_(dict)
        , all_match_(trie_->is_accepting(ROOT_STATE))
        , hit_(all_match_)
    {}

    // ── TokenAutomaton / DeadDetectable interface ───────────────────────────
    void step(Token t) noexcept {
        if (hit_) return;
        const uint8_t* data = dict_.data(t);
        const size_t   len  = dict_.token_size(t);
        for (size_t i = 0; i < len; ++i) {
            state_ = trie_->advance(state_, data[i]);
            if (trie_->is_accepting(state_)) {
                hit_ = true;
                return;
            }
        }
    }

    bool is_accepted() const noexcept { return hit_; }
    void reset()       noexcept       { state_ = ROOT_STATE; hit_ = all_match_; }
    bool is_dead()     const noexcept { return hit_; }

private:
    static constexpr State ROOT_STATE = AhoCorasickTrie::ROOT_STATE;

    std::shared_ptr<const AhoCorasickTrie> trie_;
    DictionaryView                         dict_;

    State  state_        = ROOT_STATE;
    bool   all_match_    = false;
    bool   hit_          = false;
};

} // namespace onpair::search
