#include <onpair/encoding/parsing/parser.h>
#include <onpair/encoding/parsing/bit_writer.h>

namespace onpair::encoding {

void parse(const uint8_t*              data,
           const uint32_t*             offsets,
           size_t                      n,
           const LongestPrefixMatcher& lpm,
           BitWidth                    bits,
           Store&                store)
{
    store.bit_width = bits;
    store.packed.clear();
    store.boundaries.clear();
    store.boundaries.reserve(n + 1);
    store.boundaries.push_back(0);

    BitWriter writer(store);

    for (size_t i = 0; i < n; ++i) {
        const uint8_t* str = data + offsets[i];
        const size_t   len = offsets[i + 1] - offsets[i];
        size_t pos = 0;

        while (pos < len) {
            auto m = lpm.find_longest_match(str + pos, len - pos);
            writer.write(m.first);
            pos += m.second;
        }

        store.boundaries.push_back(static_cast<uint32_t>(writer.tokens_written()));
    }

    writer.flush();
}

} // namespace onpair::encoding
