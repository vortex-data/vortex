#pragma once
#include <onpair/core/types.h>
#include <cstdint>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// TokenCursor<Bits> — pull-model iterator over a bit-packed token stream.
//
// Bits is a compile-time constant (9-16) so all masks and
// shifts fold into literals.  Resolve the runtime bit width once with
// dispatch_bits(), then work with a monomorphised cursor inside the lambda.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair::decoding {

// ─── TokenCursor ─────────────────────────────────────────────────────────────

template<BitWidth Bits>
class TokenCursor {
    static_assert(is_valid_bits(Bits), "Bits must be in [9, 16]");

    static constexpr uint32_t MASK = (uint32_t(1) << Bits) - 1;

    const uint8_t*  base_;      // byte pointer into the bit-packed buffer
    uint32_t        bit_pos_;   // current bit offset into the stream
    uint32_t        bit_end_;   // one-past-the-last bit offset

public:
    TokenCursor() noexcept = default;

    // Bind to a packed buffer without selecting a span yet.
    // Call reset_to(span) before reading.
    explicit TokenCursor(const uint64_t* ONPAIR_RESTRICT packed) noexcept
        : base_(reinterpret_cast<const uint8_t*>(packed)),
          bit_pos_(0), bit_end_(0) {}

    // Bind to a packed buffer and position on [span.begin, span.end).
    TokenCursor(const uint64_t* ONPAIR_RESTRICT packed,
                StreamSpan span) noexcept
        : base_(reinterpret_cast<const uint8_t*>(packed)),
          bit_pos_(span.begin * Bits), bit_end_(span.end * Bits) {}

    // ── Observers ─────────────────────────────────────────────────────────────
    bool     has_more()  const noexcept { return bit_pos_ < bit_end_; }
    uint32_t remaining() const noexcept { return (bit_end_ - bit_pos_) / Bits; }

    // ── Pull interface ────────────────────────────────────────────────────────

    // Decode and return the next token, then advance the cursor.
    // Precondition: has_more() == true.
    Token next() noexcept {
        uint32_t raw;
        std::memcpy(&raw, base_ + (bit_pos_ >> 3), sizeof(raw));
        Token t((raw >> (bit_pos_ & 7)) & MASK);
        bit_pos_ += Bits;
        return t;
    }

    // Decode the next token without advancing the cursor.
    // Precondition: has_more() == true.
    Token peek() const noexcept {
        uint32_t raw;
        std::memcpy(&raw, base_ + (bit_pos_ >> 3), sizeof(raw));
        return Token((raw >> (bit_pos_ & 7)) & MASK);
    }

    // ── Repositioning ─────────────────────────────────────────────────────────

    // Reset to a new span inside the same packed buffer.
    void reset_to(StreamSpan span) noexcept {
        bit_pos_ = span.begin * Bits;
        bit_end_ = span.end   * Bits;
    }

};

} // namespace onpair::decoding
