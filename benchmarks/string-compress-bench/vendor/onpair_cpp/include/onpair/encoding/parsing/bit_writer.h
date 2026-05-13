#pragma once
#include <onpair/core/store.h>

// ─────────────────────────────────────────────────────────────────────────────
// BitWriter — write-only, LSB-first bit-packing into Store.
//
// Tokens are stored least-significant-bit first across consecutive uint64_t
// words.  A token whose bits straddle a word boundary is split: the low bits
// go into the current word and the remaining high bits open the next word.
//
// The destructor flushes any partial word automatically (RAII).
//
// packed always ends with one zero sentinel word so that readers can safely
// do a 4-byte look-ahead at the last token without reading past allocated
// memory.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair::encoding {

class BitWriter {
public:
    explicit BitWriter(Store& store) noexcept
        : store_(store)
        , bits_(store.bit_width)
        , mask_((uint64_t(1) << bits_) - 1)
        , buf_(0), shift_(0), count_(0), flushed_(false)
    {
        store_.packed.clear();
        store_.packed.reserve(256);
    }

    ~BitWriter() noexcept { flush(); }

    // Append one token into the packed stream.
    // Bits are written LSB-first; tokens straddling a word boundary are split
    // across two consecutive uint64_t words automatically.
    void write(Token token) noexcept {
        buf_ |= (uint64_t(token) & mask_) << shift_;
        shift_ += bits_;
        if (shift_ >= 64) {
            store_.packed.push_back(buf_);
            shift_ -= 64;
            // Spill the bits that crossed the word boundary.
            buf_ = (uint64_t(token) & mask_) >> (bits_ - shift_);
        }
        ++count_;
    }

    // Flush the in-progress word to the store (zero-padded), then append one
    // zero sentinel word so readers can safely do a 4-byte look-ahead at the
    // last token.  Idempotent: subsequent calls are no-ops.
    // Called automatically by the destructor; safe to call manually too.
    void flush() noexcept {
        if (flushed_) return;
        if (shift_ > 0) {
            store_.packed.push_back(buf_);
            buf_   = 0;
            shift_ = 0;
        }
        if (count_ > 0) store_.packed.push_back(0);  // sentinel over-read guard
        flushed_ = true;
    }

    size_t tokens_written() const noexcept { return count_; }

private:
    Store&   store_;
    const BitWidth bits_;
    const uint64_t mask_;
    uint64_t       buf_;
    int            shift_;
    size_t         count_;
    bool           flushed_;
};

} // namespace onpair::encoding
