#pragma once
#include <onpair/core/types.h>
#include <atomic>
#include <cstring>
#include <utility>

// ─────────────────────────────────────────────────────────────────────────────
// decode_all<Bits> — maximum-speed bulk decompressor.
//
// Tokens are bit-packed into uint64_t words.  For each bit width the "natural
// group" is the smallest token count whose total bits are a multiple of 64:
//
//   Bits  Tokens/group  Words/group
//     9        64            9
//    10        32            5
//    11        64           11
//    12        16            3
//    13        64           13
//    14        32            7
//    15        64           15
//    16     (plain uint16_t array, no bit manipulation)
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair::decoding {

// ── Compile-time extraction primitives ───────────────────────────────────────

namespace detail {

/// Natural group metrics for a given bit width.
template<BitWidth Bits> struct group_traits;
template<> struct group_traits< 9> { static constexpr uint32_t tokens = 64, words =  9, subs = 4; };
template<> struct group_traits<10> { static constexpr uint32_t tokens = 32, words =  5, subs = 2; };
template<> struct group_traits<11> { static constexpr uint32_t tokens = 64, words = 11, subs = 4; };
template<> struct group_traits<12> { static constexpr uint32_t tokens = 16, words =  3, subs = 1; };
template<> struct group_traits<13> { static constexpr uint32_t tokens = 64, words = 13, subs = 4; };
template<> struct group_traits<14> { static constexpr uint32_t tokens = 32, words =  7, subs = 2; };
template<> struct group_traits<15> { static constexpr uint32_t tokens = 64, words = 15, subs = 4; };

/// Super-group metrics for the offset-aware overload.
template<BitWidth Bits> struct super_group_traits {
    static constexpr uint32_t tokens = (Bits == 14) ? 32 : 64;
    static constexpr uint32_t words  = tokens * Bits / 64;
    static constexpr uint32_t subs   = tokens / 16;
};

/// Extract one token at a compile-time-known bit position.
template<BitWidth Bits, uint32_t BitPos>
inline Token extract_one(const uint64_t* packed) noexcept {
    constexpr uint64_t MASK = (1ULL << Bits) - 1;
    constexpr uint32_t w    = BitPos / 64;
    constexpr uint32_t s    = BitPos % 64;
    if constexpr (s + Bits <= 64)
        return Token((packed[w] >> s) & MASK);
    else
        return Token(((packed[w] >> s) | (packed[w + 1] << (64 - s))) & MASK);
}

/// Extract 16 consecutive tokens starting at compile-time bit offset.
template<BitWidth Bits, uint32_t StartBit, size_t... Is>
inline void extract16_impl(const uint64_t* packed, Token* out,
                           std::index_sequence<Is...>) noexcept {
    ((out[Is] = extract_one<Bits, StartBit + uint32_t(Is) * Bits>(packed)), ...);
}

template<BitWidth Bits, uint32_t StartBit = 0>
inline void extract16(const uint64_t* packed, Token* out) noexcept {
    extract16_impl<Bits, StartBit>(packed, out, std::make_index_sequence<16>{});
}

} // namespace detail

// ── decode_all (no offsets) ──────────────────────────────────────────────────

template<BitWidth Bits>
size_t decode_all(
    const uint64_t* ONPAIR_RESTRICT packed,
    const uint8_t* ONPAIR_RESTRICT dict_bytes,
    const uint32_t* ONPAIR_RESTRICT dict_offsets,
    uint32_t                     total_tokens,
    uint8_t* ONPAIR_RESTRICT out) noexcept
{
    uint8_t* const out_start = out;

    // Macro to guarantee purely inline token emission without capture overhead
    #define ONPAIR_EMIT_NO_OFF(t) do { \
        const uint32_t off = dict_offsets[(t)]; \
        const uint32_t len = dict_offsets[(t) + 1] - off; \
        std::memcpy(out, dict_bytes + off, MAX_TOKEN_SIZE); \
        out += len; \
    } while(0)

    if constexpr (Bits == 16) {
        const auto* tokens = reinterpret_cast<const uint16_t*>(packed);
        for (uint32_t i = 0; i < total_tokens; ++i) {
            ONPAIR_EMIT_NO_OFF(Token(tokens[i]));
        }
    } else {
        using G = detail::group_traits<Bits>;
        constexpr uint32_t B = Bits;

        const uint32_t full_g = total_tokens / G::tokens;
        const uint32_t rem    = total_tokens % G::tokens;

        #define ONPAIR_EMIT16_NO_OFF(tp) do { \
            for (int j = 0; j < 16; ++j) { ONPAIR_EMIT_NO_OFF((tp)[j]); } \
        } while(0)

        for (uint32_t g = 0; g < full_g; ++g, packed += G::words) {
            Token t[16];
                                        detail::extract16<Bits,  0 * B>(packed, t); ONPAIR_EMIT16_NO_OFF(t);
            if constexpr (G::subs >= 2) { detail::extract16<Bits, 16 * B>(packed, t); ONPAIR_EMIT16_NO_OFF(t); }
            if constexpr (G::subs >= 3) { detail::extract16<Bits, 32 * B>(packed, t); ONPAIR_EMIT16_NO_OFF(t); }
            if constexpr (G::subs >= 4) { detail::extract16<Bits, 48 * B>(packed, t); ONPAIR_EMIT16_NO_OFF(t); }
        }

        if (rem) {
            constexpr uint64_t M = (1ULL << Bits) - 1;
            const auto* base = reinterpret_cast<const uint8_t*>(packed);
            size_t bit_pos = 0;
            for (uint32_t i = 0; i < rem; ++i) {
                uint32_t raw;
                std::memcpy(&raw, base + (bit_pos >> 3), sizeof(raw));
                ONPAIR_EMIT_NO_OFF(Token((raw >> (bit_pos & 7)) & M));
                bit_pos += Bits;
            }
        }
        
        #undef ONPAIR_EMIT16_NO_OFF
    }
    
    #undef ONPAIR_EMIT_NO_OFF

    return size_t(out - out_start);
}

// ── decode_all (with Arrow-style offsets) ────────────────────────────────────

template<BitWidth Bits>
size_t decode_all(
    const uint64_t* ONPAIR_RESTRICT packed,
    const uint32_t* ONPAIR_RESTRICT boundaries,
    const uint8_t*  ONPAIR_RESTRICT dict_bytes,
    const uint32_t* ONPAIR_RESTRICT dict_offsets,
    uint32_t                     total_tokens,
    size_t                       total_strings,
    uint8_t*        ONPAIR_RESTRICT out,
    uint32_t*       ONPAIR_RESTRICT out_offsets) noexcept
{
    // ── 16-bit: iterate per string (no bit manipulation) ─────────────────
    if constexpr (Bits == 16) {
        uint8_t* const out_start = out;
        const auto* tokens = reinterpret_cast<const uint16_t*>(packed);
        for (size_t s = 0; s < total_strings; ++s) {
            out_offsets[s] = static_cast<uint32_t>(out - out_start);
            const uint32_t tk_end = boundaries[s + 1];
            for (uint32_t i = boundaries[s]; i < tk_end; ++i) {
                const uint32_t off = dict_offsets[tokens[i]];
                const uint32_t len = dict_offsets[tokens[i] + 1] - off;
                std::memcpy(out, dict_bytes + off, MAX_TOKEN_SIZE);
                out += len;
            }
        }
        out_offsets[total_strings] = static_cast<uint32_t>(out - out_start);
        return size_t(out - out_start);
    }

    // ── Bit-packed paths (9–15 bit) ──────────────────────────────────────
    // Tokens are extracted in groups tied to word boundaries.  We cannot
    // iterate per-string, so we emit all tokens in a group first (Phase 1)
    // then compute byte positions via prefix-sum (Phase 2) and resolve
    // which string boundaries fell within this group.
    else {
        // When total_tokens == 0 all strings are empty and neither the main loop
        // nor the remainder block runs, so resolve_boundaries is never called and
        // out_offsets would be left uninitialised. Handle it explicitly here.
        if (total_tokens == 0) {
            for (size_t i = 0; i <= total_strings; ++i)
                out_offsets[i] = 0;
            return 0;
        }

        using SG = detail::super_group_traits<Bits>;
        constexpr uint32_t B = Bits;

        uint8_t* const out_start = out;
        uint32_t current_string  = 0;
        uint32_t tk_start        = 0;
        uint32_t sg_offsets[SG::tokens + 1];

        // Core extraction macros
        #define ONPAIR_EMIT(t, local_idx) do { \
            sg_offsets[(local_idx)] = static_cast<uint32_t>(out - out_start); \
            const uint32_t off = dict_offsets[(t)]; \
            const uint32_t len = dict_offsets[(t) + 1] - off; \
            std::memcpy(out, dict_bytes + off, MAX_TOKEN_SIZE); \
            out += len; \
        } while(0)

        // Emit a batch of 4 tokens with a compiler barrier after, to prevent
        // the register allocator from trying to keep too many intermediate
        // output pointers live simultaneously.
        #define ONPAIR_EMIT4(tp, base) do { \
            ONPAIR_EMIT((tp)[0], (base)); \
            ONPAIR_EMIT((tp)[1], (base) + 1); \
            ONPAIR_EMIT((tp)[2], (base) + 2); \
            ONPAIR_EMIT((tp)[3], (base) + 3); \
            std::atomic_signal_fence(std::memory_order_acq_rel); \
        } while(0)

        #define ONPAIR_RESOLVE_BOUNDARIES(tk_count) do { \
            sg_offsets[(tk_count)] = static_cast<uint32_t>(out - out_start); \
            const uint32_t tk_end = tk_start + (tk_count); \
            while (current_string <= total_strings && boundaries[current_string] <= tk_end) { \
                out_offsets[current_string] = sg_offsets[boundaries[current_string] - tk_start]; \
                ++current_string; \
            } \
            tk_start = tk_end; \
        } while(0)

        const uint32_t full_sg = total_tokens / SG::tokens;
        const uint32_t rem_sg  = total_tokens % SG::tokens;

        for (uint32_t g = 0; g < full_sg; ++g, packed += SG::words) {
            Token t[16];

            detail::extract16<Bits, 0>(packed, t);
            ONPAIR_EMIT4(t, 0); ONPAIR_EMIT4(t + 4, 4); ONPAIR_EMIT4(t + 8, 8); ONPAIR_EMIT4(t + 12, 12);

            if constexpr (SG::subs >= 2) {
                detail::extract16<Bits, 16 * B>(packed, t);
                ONPAIR_EMIT4(t, 16); ONPAIR_EMIT4(t + 4, 20); ONPAIR_EMIT4(t + 8, 24); ONPAIR_EMIT4(t + 12, 28);
            }
            if constexpr (SG::subs >= 3) {
                detail::extract16<Bits, 32 * B>(packed, t);
                ONPAIR_EMIT4(t, 32); ONPAIR_EMIT4(t + 4, 36); ONPAIR_EMIT4(t + 8, 40); ONPAIR_EMIT4(t + 12, 44);
            }
            if constexpr (SG::subs >= 4) {
                detail::extract16<Bits, 48 * B>(packed, t);
                ONPAIR_EMIT4(t, 48); ONPAIR_EMIT4(t + 4, 52); ONPAIR_EMIT4(t + 8, 56); ONPAIR_EMIT4(t + 12, 60);
            }

            ONPAIR_RESOLVE_BOUNDARIES(SG::tokens);
        }

        if (rem_sg) {
            constexpr uint64_t M = (1ULL << Bits) - 1;
            const auto* base = reinterpret_cast<const uint8_t*>(packed);
            size_t bit_pos = 0;
            for (uint32_t i = 0; i < rem_sg; ++i) {
                uint32_t raw;
                std::memcpy(&raw, base + (bit_pos >> 3), sizeof(raw));
                ONPAIR_EMIT(Token((raw >> (bit_pos & 7)) & M), i);
                bit_pos += Bits;
            }
            ONPAIR_RESOLVE_BOUNDARIES(rem_sg);
        }

        // Clean up macros so they don't leak out of the file
        #undef ONPAIR_EMIT
        #undef ONPAIR_EMIT4
        #undef ONPAIR_RESOLVE_BOUNDARIES

        return size_t(out - out_start);
    }
}

} // namespace onpair::decoding
