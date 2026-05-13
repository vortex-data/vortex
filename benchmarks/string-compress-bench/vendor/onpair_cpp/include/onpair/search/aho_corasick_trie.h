#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// AhoCorasickTrie
// ─────────────────────────────────────────────────────────────────────────────
// Byte-level Aho-Corasick trie with failure links for multi-pattern matching.
// Used by AhoCorasickAutomaton to build its token-level transition table.
//
// Precondition: the combined patterns must produce strictly fewer than 65 535
// trie nodes (UINT16_MAX is reserved as sentinel during construction).  In
// the worst case (no shared prefixes) this is the sum of all pattern lengths
// plus one (root).  Exceeding this limit is undefined behaviour.

class AhoCorasickTrie {
public:
    using State = uint16_t;

    static constexpr State ROOT_STATE = 0;
    static constexpr State NULL_STATE = UINT16_MAX;

    explicit AhoCorasickTrie(std::span<const std::string_view> patterns);

    // Advances by one byte, automatically resolving failure links.
    State advance(State u, uint8_t c) const noexcept {
        while (true) {
            const uint16_t start = child_offsets_[u];
            const uint16_t end   = child_offsets_[u + 1];

            for (uint16_t i = start; i < end; ++i) {
                if (edge_labels_[i] == c) return edge_targets_[i];
                if (edge_labels_[i] > c) break;
            }

            if (u == ROOT_STATE) return ROOT_STATE;
            u = fail_[u];
        }
    }

    // Accessors
    bool   is_accepting(State s) const noexcept { return is_accepting_[s]; }
    size_t num_states()          const noexcept { return num_states_; }
    size_t num_patterns()        const noexcept { return num_patterns_; }

    std::span<const uint8_t> edge_labels(State state) const noexcept {
        return {edge_labels_.data() + child_offsets_[state],
                edge_labels_.data() + child_offsets_[state + 1]};
    }

    std::span<const State> edge_targets(State state) const noexcept {
        return {edge_targets_.data() + child_offsets_[state],
                edge_targets_.data() + child_offsets_[state + 1]};
    }

    State fail_link(State s) const noexcept { return fail_[s]; }

private:
    std::vector<uint8_t>  edge_labels_;
    std::vector<State>    edge_targets_;
    std::vector<uint16_t> child_offsets_;
    std::vector<State>    fail_;
    std::vector<bool>     is_accepting_;

    size_t num_patterns_  = 0;
    size_t num_states_ = 0;
};

// ─── Implementation ─────────────────────────────────────────────────────────

inline AhoCorasickTrie::AhoCorasickTrie(std::span<const std::string_view> patterns) {
    num_patterns_ = patterns.size();

    // Temporary structure: First-Child / Next-Sibling
    struct TrieNode {
        uint8_t c = 0;
        State   first_child  = NULL_STATE;
        State   next_sibling = NULL_STATE;
    };

    std::vector<TrieNode> nodes(1); // nodes[0] is ROOT_STATE
    is_accepting_.push_back(false); 

    // ── 1. Build Trie (with inline sorted insertion) ────────────────────────
    for (const auto& pat : patterns) {
        if (pat.empty()) {
            is_accepting_[ROOT_STATE] = true;
            continue;
        }
        
        const auto* p = reinterpret_cast<const uint8_t*>(pat.data());
        State cur = ROOT_STATE;
        
        for (size_t i = 0; i < pat.size(); ++i) {
            State child = nodes[cur].first_child;
            State prev  = NULL_STATE;
            
            // Traverse the sibling list, stopping if we find the byte or 
            // pass where it should be
            while (child != NULL_STATE && nodes[child].c < p[i]) {
                prev  = child;
                child = nodes[child].next_sibling;
            }

            // If the branch doesn't exist, insert it maintaining sorted order
            if (child == NULL_STATE || nodes[child].c != p[i]) {
                State new_node = static_cast<State>(nodes.size());
                
                // next_sibling points to 'child' to maintain alphabetical order
                nodes.push_back({p[i], NULL_STATE, child}); 
                is_accepting_.push_back(false);
                
                if (prev == NULL_STATE) {
                    nodes[cur].first_child = new_node;
                } else {
                    nodes[prev].next_sibling = new_node;
                }
                child = new_node;
            }
            cur = child;
        }
        is_accepting_[cur] = true;
    }

    num_states_ = nodes.size();

    // ── 2. Compact into SoA format ──────────────────────────────────────────
    child_offsets_.reserve(num_states_ + 1);
    edge_labels_.reserve(num_states_);
    edge_targets_.reserve(num_states_);

    for (State i = 0; i < num_states_; ++i) {
        child_offsets_.push_back(static_cast<uint16_t>(edge_labels_.size()));
        
        // Edges are already sorted by label due to the insertion method, 
        // so we can just append them
        State child = nodes[i].first_child;
        while (child != NULL_STATE) {
            edge_labels_.push_back(nodes[child].c);
            edge_targets_.push_back(child);
            child = nodes[child].next_sibling;
        }
    }
    child_offsets_.push_back(static_cast<uint16_t>(edge_labels_.size()));

    // ── 3. Compute failure links via BFS ────────────────────────────────────
    fail_.assign(num_states_, ROOT_STATE);
    std::vector<State> bfs;
    bfs.reserve(num_states_);

    for (State target : edge_targets(ROOT_STATE)) {
        fail_[target] = ROOT_STATE;
        bfs.push_back(target);
    }

    for (size_t qi = 0; qi < bfs.size(); ++qi) {
        State u = bfs[qi];

        if (is_accepting_[fail_[u]]) {
            is_accepting_[u] = true;
        }

        auto labels  = edge_labels(u);
        auto targets = edge_targets(u);
        for (uint16_t i = 0; i < labels.size(); ++i) {
            fail_[targets[i]] = advance(fail_[u], labels[i]);
            bfs.push_back(targets[i]);
        }
    }
}

} // namespace onpair::search