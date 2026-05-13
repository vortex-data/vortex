#pragma once
#include <onpair/core/dictionary.h>
#include <onpair/encoding/training/config.h>
#include <onpair/encoding/lpm.h>
#include <cstdint>
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// Trainer — encoding-internal API.
//
// train()
//   Runs the OnPair pair-discovery algorithm over the input strings and builds
//   a dictionary of up to 2^cfg.bits tokens.  The first 256 tokens are always
//   the single-byte values 0x00–0xFF; subsequent tokens are pair merges
//   discovered during the training scan.
//
//   The returned dictionary is always sorted lexicographically by token byte
//   sequence, with token IDs reassigned to match the sorted order.  Sorting
//   is performed as the final step of training and enables optimised query
//   operations (e.g. binary-search prefix range lookups) on the compressed
//   representation.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair::encoding {

/// Result of train(): a sorted dictionary and a matching longest-prefix
/// matcher whose token IDs correspond to the dictionary's sorted order.
struct TrainResult {
    Dictionary             dict;  /// tokens in lexicographic order
    LongestPrefixMatcher   lpm;   /// maps byte sequences → sorted token IDs
};

/// Train a dictionary from raw input and return it in sorted order.
TrainResult train(const uint8_t* data,
                  const uint32_t* offsets,
                  size_t n,
                  const TrainingConfig& cfg);

} // namespace onpair::encoding
