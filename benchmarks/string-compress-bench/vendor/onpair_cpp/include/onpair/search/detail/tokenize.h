#pragma once
#include <onpair/core/types.h>
#include <onpair/core/dictionary_view.h>
#include <cstddef>
#include <cstring>
#include <string_view>
#include <vector>

namespace onpair::search::detail {

// ─────────────────────────────────────────────────────────────────────────────
// tokenize
// ─────────────────────────────────────────────────────────────────────────────
// Greedy longest-match tokenisation of `text` against a dictionary.
//
// Precondition: the dictionary must be sorted and contain the 256 single-byte 
// base tokens.

inline std::vector<Token> tokenize(std::string_view text,
                                   DictionaryView dv)
{
    std::vector<Token> tokens;
    tokens.reserve(text.size());

    const auto*     data    = reinterpret_cast<const uint8_t*>(text.data());
    const uint8_t*  bytes   = dv.raw_bytes();
    const uint32_t* offsets = dv.raw_offsets();

    auto tlen = [&](Token t) -> uint32_t {
        return offsets[t + 1] - offsets[t];
    };
    auto byte_at = [&](Token t, size_t k) -> uint8_t {
        return bytes[offsets[t] + k];
    };

    size_t pos = 0;

    while (pos < text.size()) {
        const size_t remaining = text.size() - pos;
        const size_t max_len   = remaining < MAX_TOKEN_SIZE
                               ? remaining : MAX_TOKEN_SIZE;

        Token      best  = 0;
        TokenRange range = {0, static_cast<Token>(dv.num_tokens() - 1)};

        for (size_t k = 0; k < max_len; ++k) {
            const uint8_t target = data[pos + k];

            // Lower bound: first token in range with byte[k] >= target.
            // Tokens shorter than k+1 sort before any that has the byte.
            Token lo = range.begin, hi = range.last;
            while (lo < hi) {
                Token mid = lo + ((hi - lo) >> 1);
                if (tlen(mid) <= k || byte_at(mid, k) < target)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (tlen(lo) <= k || byte_at(lo, k) != target) break;

            Token first = lo;

            // Upper bound: find the first token with byte[k] > target,
            // then step back to get the last with byte[k] == target.
            lo = first;
            hi = range.last;
            while (lo < hi) {
                Token mid = lo + ((hi - lo) >> 1);
                if (tlen(mid) <= k || byte_at(mid, k) <= target)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            Token last = (tlen(lo) > k && byte_at(lo, k) > target)
                       ? Token(lo - 1) : lo;

            // Shortest token in range comes first in lexicographic order.
            // If it has length exactly k+1, it is an exact match.
            if (tlen(first) == k + 1)
                best = first;

            range = {first, last};
        }

        tokens.push_back(best);
        pos += tlen(best);
    }

    return tokens;
}

} // namespace onpair::search::detail
