#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// OnPair — the only header a user needs to include.
//
// Usage:
//   #include <onpair/api.h>
//
// Key types:
//   onpair::OnPairColumn     — owning compressed column
//   onpair::OnPairColumnView — non-owning view for read ops (decompress, search)
//   onpair::encoding::TrainingConfig  — compression settings
//   onpair::encoding::DynamicThreshold / FixedThreshold — threshold policies
//   onpair::search::KmpAutomaton      — low-level substring automaton
//   onpair::DECOMPRESS_BUFFER_PADDING — padding for decompress() output buffer
//
// Example:
//   #include <onpair/api.h>
//   namespace op = onpair;
//
//   op::OnPairColumn col = op::OnPairColumn::compress(strings);
//   auto view = col.view();
//
//   std::vector<char> buf(max_len + op::DECOMPRESS_BUFFER_PADDING);
//   size_t len = view.decompress(42, buf.data());
//
//   auto hits = view.contains("needle");
// ─────────────────────────────────────────────────────────────────────────────

// Column types
#include <onpair/column/column.h>

// Compression configuration
#include <onpair/encoding/training/config.h>

// Search primitives (for low-level scan API)
#include <onpair/search/automata/aho_corasick_automaton.h>
#include <onpair/search/automata/eq_automaton.h>
#include <onpair/search/automata/kmp_automaton.h>
#include <onpair/search/automata/prefix_automaton.h>