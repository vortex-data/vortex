#pragma once
#include <onpair/core/dictionary_view.h>
#include <onpair/core/types.h>
#include <boost/unordered/unordered_flat_map.hpp>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// LongestPrefixMatcher — encoding-internal data structure.
//
// Maps byte sequences to Token IDs and, given an input buffer, finds the
// longest token whose byte sequence is a prefix of that buffer.
//
// ── Construction ──────────────────────────────────────────────────────────────
//
//   LongestPrefixMatcher()
//     Default constructor.  Pre-inserts all 256 single-byte tokens with IDs
//     0–255.  After construction find_longest_match always returns a valid
//     result; size() == 256.  Use this to begin a training session.
//
//   LongestPrefixMatcher::from_dictionary(DictionaryView dict)
//     Static factory.  Builds a matcher whose tokens and IDs correspond
//     exactly to the entries in `dict` (token at position i receives ID i).
//     Precondition: dict must be a complete, valid OnPair dictionary — in
//     particular it must contain all 256 single-byte tokens so that
//     find_longest_match always succeeds.
//
// ── Token ID assignment ───────────────────────────────────────────────────────
//
//   insert() assigns the next available Token ID and returns it.  The counter
//   starts at 256 after default construction and at dict.num_tokens() after
//   from_dictionary().
//   Precondition: size() must be strictly less than the Token ID space capacity
//   (2^16 = 65 536 for the current uint16_t Token type).  Exceeding this limit
//   produces a duplicate ID and corrupts the matcher — behaviour is undefined.
//
// ── find_longest_match ────────────────────────────────────────────────────────
//
//   Precondition: the matcher must contain all 256 single-byte tokens, which
//   is guaranteed by the default constructor and by from_dictionary() when
//   given a complete dictionary.  Violating this precondition on input that
//   has no match results in undefined behaviour.
//
// ── Storage strategy ──────────────────────────────────────────────────────────
//
//   Short patterns (1–8 bytes):
//     Direct hash lookup: (bytes_as_uint64, length) → Token
//   Long patterns (9–MAX_TOKEN_SIZE bytes):
//     Bucketed by their 8-byte prefix.  Each bucket starts as a LinearBucket
//     (sorted vector) and is promoted to a TrieBucket when the entry count 
//     exceeds PROMOTE_THRESHOLD.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair::encoding {

class LongestPrefixMatcher {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    // Pre-inserts all 256 single-byte tokens (IDs 0–255).
    // Postcondition: size() == 256; find_longest_match always returns a match.
    LongestPrefixMatcher() : LongestPrefixMatcher(from_dict_tag{}) {
        for (uint16_t i = 0; i <= 255; ++i) {
            const uint8_t b = static_cast<uint8_t>(i);
            insert_internal(&b, 1, Token(i));
        }
        next_id_ = 256;
    }

    // Builds a matcher from a complete, pre-existing dictionary.
    // Token IDs correspond to dictionary positions: token at index i → ID i.
    //
    // Precondition: dict is a complete OnPair dictionary containing all 256
    //   single-byte tokens and at most 2^16 entries in total.
    // Postcondition: size() == dict.num_tokens(); find_longest_match always
    //   returns a match.
    static LongestPrefixMatcher from_dictionary(DictionaryView dict) {
        LongestPrefixMatcher lpm(from_dict_tag{});
        const size_t n = dict.num_tokens();
        for (size_t i = 0; i < n; ++i) {
            const Token t = Token(i);
            lpm.insert_internal(dict.data(t), dict.token_size(t), t);
        }
        lpm.next_id_ = uint32_t(n);
        return lpm;
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    // Inserts the byte sequence [data, data+length) and assigns it the next
    // available Token ID.  Returns the assigned ID.
    //
    // Preconditions:
    //   - data is a valid pointer to at least `length` readable bytes.
    //   - 1 <= length <= MAX_TOKEN_SIZE.
    //   - size() < 65 536  (Token ID space must not be exhausted).
    // Postcondition: size() increases by one; find_longest_match will return
    //   the new token for any input whose longest matching prefix is this one.
    Token insert(const uint8_t* data, size_t length) {
        const Token id = Token(next_id_++);
        insert_internal(data, length, id);
        return id;
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    // Returns the token whose byte sequence is the longest prefix of
    // [data, data+length), together with that prefix length.
    //
    // Preconditions:
    //   - data is a valid pointer to at least `length` readable bytes.
    //   - length >= 1.
    //   - All 256 single-byte tokens are present (guaranteed by the default
    //     constructor and from_dictionary() with a complete dictionary).
    // Postcondition: returned length is in [1, min(length, MAX_TOKEN_SIZE)].
    std::pair<Token, size_t>
    find_longest_match(const uint8_t* data, size_t length) const noexcept {
        if (length > BUCKET_PREFIX_LEN) {
            const uint64_t prefix = load8(data);
            const auto bit = long_.find(prefix);

            if (bit != long_.end()) {
                const size_t slen =
                    std::min(length, MAX_TOKEN_SIZE) - BUCKET_PREFIX_LEN;

                auto result = std::visit(
                    [&](const auto& b) {
                        return b.search_suffix(
                            data + BUCKET_PREFIX_LEN, slen, pool_);
                    },
                    bit->second);

                if (result.has_value())
                    return { result->first, BUCKET_PREFIX_LEN + result->second };
            }
        }

        const size_t   limit = std::min(length, BUCKET_PREFIX_LEN);
        const uint64_t val   = load_le(data, limit);
        for (size_t len = limit; len > 0; --len) {
            const auto it =
                short_.find(std::make_pair(masked(val, len), uint8_t(len)));
            if (it != short_.end())
                return { it->second, len };
        }

        // Unreachable when all single-byte tokens are present (precondition).
        unreachable();
    }

    // Number of tokens currently mapped.
    // Equal to 256 after default construction; grows by one per insert() call.
    // Stored as uint32_t to avoid wrapping when the full 16-bit Token space
    // (65 536 entries) is used.
    size_t size() const noexcept { return next_id_; }

private:
    // ── Constants ────────────────────────────────────────────────────────────
    static constexpr size_t   BUCKET_PREFIX_LEN = 8;
    static constexpr size_t   PROMOTE_THRESHOLD = 128;
    static constexpr uint32_t INVALID_IDX       = UINT32_MAX;

    static constexpr uint64_t MASKS[9] = {
        0x0000000000000000ULL, 0x00000000000000FFULL, 0x000000000000FFFFULL,
        0x0000000000FFFFFFULL, 0x00000000FFFFFFFFULL, 0x000000FFFFFFFFFFULL,
        0x0000FFFFFFFFFFFFULL, 0x00FFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    };

    // ── Types ─────────────────────────────────────────────────────────────────
    using Entry = std::tuple<uint64_t, uint8_t, Token>;

    struct TrieNode {
        std::optional<Token>                      id;
        std::vector<std::pair<uint8_t, uint32_t>> children;
    };

    using Pool = std::vector<TrieNode>;

    // ── Tag for the pool-only constructor used by from_dictionary ─────────────
    struct from_dict_tag {};

    explicit LongestPrefixMatcher(from_dict_tag) : next_id_(0) {
        pool_.reserve(256 * 1024);
    }

    // ── Pool helpers ─────────────────────────────────────────────────────────
    static uint32_t alloc_node(Pool& pool) {
        const uint32_t idx = static_cast<uint32_t>(pool.size());
        pool.emplace_back();
        return idx;
    }

    static uint32_t find_child(const Pool& pool, uint32_t node,
                               uint8_t byte) noexcept {
        for (const auto& [b, idx] : pool[node].children)
            if (b == byte) return idx;
        return INVALID_IDX;
    }

    // ── Bucket types ─────────────────────────────────────────────────────────
    //
    // LinearBucket and TrieBucket share the same interface:
    //   insert_suffix(suf, slen, id, pool)
    //   search_suffix(suf, max_slen, pool) → optional<{id, matched_len}>
    //
    // std::visit dispatches uniformly; callers never inspect the active type.

    struct LinearBucket {
        std::vector<Entry> entries;

        void insert_suffix(const uint8_t* suf, size_t slen, Token id,
                           Pool& /*pool*/) {
            entries.emplace_back(load_le(suf, slen), uint8_t(slen), id);
            std::sort(entries.begin(), entries.end(),
                      [](const Entry& a, const Entry& b) {
                          return std::get<1>(b) < std::get<1>(a); // descending length
                      });
        }

        std::optional<std::pair<Token, size_t>>
        search_suffix(const uint8_t* suf, size_t max_slen,
                      const Pool& /*pool*/) const noexcept {
            const uint64_t val = load_le(suf, max_slen);
            for (const auto& [esuf, elen, eid] : entries) {
                if (elen <= max_slen &&
                    (std::countr_zero(val ^ esuf) >> 3) >= elen)
                    return std::make_pair(eid, size_t(elen));
            }
            return std::nullopt;
        }
    };

    struct TrieBucket {
        uint32_t root;

        explicit TrieBucket(uint32_t r) : root(r) {}

        void insert_suffix(const uint8_t* suf, size_t slen, Token id,
                           Pool& pool) {
            uint32_t cur = root;
            for (size_t i = 0; i < slen; ++i) {
                uint32_t child = find_child(pool, cur, suf[i]);
                if (child == INVALID_IDX) {
                    const uint32_t new_idx = alloc_node(pool);
                    pool[cur].children.emplace_back(suf[i], new_idx);
                    cur = new_idx;
                } else {
                    cur = child;
                }
            }
            pool[cur].id = id;
        }

        std::optional<std::pair<Token, size_t>>
        search_suffix(const uint8_t* suf, size_t max_slen,
                      const Pool& pool) const noexcept {
            std::optional<std::pair<Token, size_t>> best;
            uint32_t cur = root;
            for (size_t pos = 0; pos < max_slen; ++pos) {
                const uint32_t child = find_child(pool, cur, suf[pos]);
                if (child == INVALID_IDX) break;
                cur = child;
                if (pool[cur].id.has_value())
                    best = { *pool[cur].id, pos + 1 };
            }
            return best;
        }
    };

    using Bucket = std::variant<LinearBucket, TrieBucket>;

    // ── Internal insert ───────────────────────────────────────────────────────
    // Routes the byte sequence to the short or long store and, if the long
    // bucket has grown past PROMOTE_THRESHOLD, promotes it to a TrieBucket.
    void insert_internal(const uint8_t* data, size_t length, Token id) {
        if (length <= BUCKET_PREFIX_LEN) {
            const uint64_t key = load_le(data, length);
            short_.emplace(std::make_pair(key, uint8_t(length)), id);
            return;
        }

        const uint64_t prefix = load8(data);
        auto& bucket = long_[prefix];

        const uint8_t* suffix = data + BUCKET_PREFIX_LEN;
        const size_t   slen   = length - BUCKET_PREFIX_LEN;

        std::visit([&](auto& b) { b.insert_suffix(suffix, slen, id, pool_); },
                   bucket);

        if (auto* lb = std::get_if<LinearBucket>(&bucket)) {
            if (lb->entries.size() > PROMOTE_THRESHOLD)
                bucket = promote(*lb);
        }
    }

    // ── Bit-manipulation helpers ──────────────────────────────────────────────
    static uint64_t load_le(const uint8_t* p, size_t len) noexcept {
        uint64_t v = 0;
        std::memcpy(&v, p, len);
        return v;
    }

    static uint64_t load8(const uint8_t* p) noexcept {
        uint64_t v;
        std::memcpy(&v, p, 8);
        return v;
    }

    static uint64_t masked(uint64_t v, size_t len) noexcept {
        return v & MASKS[len];
    }

    // ── LinearBucket → TrieBucket promotion ──────────────────────────────────
    TrieBucket promote(LinearBucket& lb) {
        TrieBucket tb{alloc_node(pool_)};
        for (const auto& [suffix, slen, id] : lb.entries) {
            uint8_t buf[8];
            std::memcpy(buf, &suffix, sizeof(suffix));
            tb.insert_suffix(buf, slen, id, pool_);
        }
        return tb;
    }

    // ── Storage ───────────────────────────────────────────────────────────────
    boost::unordered_flat_map<std::pair<uint64_t, uint8_t>, Token> short_;
    boost::unordered_flat_map<uint64_t, Bucket>                    long_;
    Pool     pool_;
    uint32_t next_id_ = 0; // uint32_t avoids wrap-around when all 65 536 Token IDs are used
};

} // namespace onpair::encoding
