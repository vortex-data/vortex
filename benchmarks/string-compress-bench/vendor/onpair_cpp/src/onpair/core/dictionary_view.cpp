#include <onpair/core/dictionary_view.h>

namespace onpair {

TokenRange DictionaryView::prefix_range(
    const uint8_t* prefix, size_t plen) const noexcept
{
    // A prefix larger than any possible token can never match anything.
    if (plen > MAX_TOKEN_SIZE)
        return TokenRange{};

    const uint8_t*  bytes   = dict_.bytes.data();
    const uint32_t* offsets = dict_.offsets.data();
    const uint32_t  n       = static_cast<uint32_t>(dict_.num_tokens());

    // Binary search for the first token whose bytes >= [target, tlen),
    // restricted to the half-open index range [start, n).
    auto lower_bound = [&](const uint8_t* target, size_t tlen,
                            uint32_t start) -> uint32_t
    {
        uint32_t lo = start, hi = n;
        while (lo < hi) {
            const uint32_t mid  = lo + ((hi - lo) >> 1);
            const uint32_t moff = offsets[mid];
            const uint32_t mlen = offsets[mid + 1] - moff;
            const size_t   clen = mlen < tlen ? mlen : tlen;
            const int      cmp  = std::memcmp(bytes + moff, target, clen);

            // token[mid] < target  iff  cmp < 0,  or cmp == 0 and token is shorter
            if (cmp < 0 || (cmp == 0 && mlen < tlen))
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    };

    // First search: find lo from the beginning of the dictionary.
    const uint32_t lo = lower_bound(prefix, plen, 0);

    // Compute the next lexicographic prefix by incrementing the last
    // non-0xFF byte, trimming trailing 0xFF bytes first.
    uint8_t buf[MAX_TOKEN_SIZE];
    size_t  ulen     = plen;
    bool    overflow = true;

    while (ulen > 0) {
        if (prefix[ulen - 1] < 0xFF) {
            std::memcpy(buf, prefix, ulen);
            buf[ulen - 1]++;
            overflow = false;
            break;
        }
        --ulen;
    }

    // Second search: hi >= lo always, so start from lo, not from 0.
    const uint32_t hi = overflow ? n : lower_bound(buf, ulen, lo);

    return lo < hi ? TokenRange{ Token(lo), Token(hi - 1) }
                   : TokenRange{};
}

} // namespace onpair
