#pragma once
#include <onpair/core/store.h>
#include <onpair/core/types.h>
#include <onpair/encoding/lpm.h>
#include <cstdint>
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// Parser — encoding-internal API.
//
// parse() drives the LongestPrefixMatcher over every input string, writes the
// resulting token IDs into a Store via BitWriter, and records per-string
// token-count boundaries.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair::encoding {

// Encode all strings into `store` using `lpm`.
// data[offsets[i]..offsets[i+1]) is string i; offsets has n+1 elements.
void parse(const uint8_t*              data,
           const uint32_t*             offsets,
           size_t                      n,
           const LongestPrefixMatcher& lpm,
           BitWidth                    bits,
           Store&                store);

} // namespace onpair::encoding
