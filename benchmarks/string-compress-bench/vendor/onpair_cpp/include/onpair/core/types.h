#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#ifndef ONPAIR_RESTRICT
#  if defined(_MSC_VER)
#    define ONPAIR_RESTRICT __restrict
#  elif defined(__clang__) || defined(__GNUC__)
#    define ONPAIR_RESTRICT __restrict__
#  else
#    define ONPAIR_RESTRICT
#  endif
#endif

namespace onpair {

using BitWidth = uint8_t;   // Legal values: 9–16
using Token    = uint16_t;

// Maximum byte size of any token.
inline constexpr size_t MAX_TOKEN_SIZE = 16;

// Range of bytes [begin, end) inside the dictionary buffer.
struct ByteSpan  { uint32_t begin, end; uint32_t size() const noexcept { return end - begin; } };

// Range of positions [begin, end) inside the packed store.
struct StreamSpan { uint32_t begin, end; uint32_t size() const noexcept { return end - begin; } };

// Closed range of token IDs [begin, last].
// Default {1, 0} is the canonical empty range (begin > last).
struct TokenRange {
    Token begin = 1;  // first token id (inclusive)
    Token last  = 0;  // last  token id (inclusive); begin > last => empty

    bool     empty()    const noexcept { return begin > last; }
    uint32_t size()     const noexcept {
        return empty() ? 0u : uint32_t(last) - uint32_t(begin) + 1u;
    }
    bool     contains(Token t) const noexcept { return t >= begin && t <= last; }
};

constexpr size_t max_dict_size(BitWidth bits) noexcept { return size_t(1) << bits; }
constexpr bool   is_valid_bits(BitWidth b)    noexcept { return b >= 9 && b <= 16; }

[[noreturn]] inline void unreachable() noexcept {
#if defined(_MSC_VER)
    __assume(false);
#elif defined(__clang__) || defined(__GNUC__)
    __builtin_unreachable();
#else
    std::abort();
#endif
}

// Resolve a runtime BitWidth to a compile-time constant and invoke `fn`.
// `fn` receives a std::integral_constant<BitWidth, N> whose ::value is usable
// as a template argument.
template<typename F>
decltype(auto) dispatch_bits(BitWidth bw, F&& fn) {
    switch (bw) {
        case 9:  return fn(std::integral_constant<BitWidth, 9>{});
        case 10: return fn(std::integral_constant<BitWidth, 10>{});
        case 11: return fn(std::integral_constant<BitWidth, 11>{});
        case 12: return fn(std::integral_constant<BitWidth, 12>{});
        case 13: return fn(std::integral_constant<BitWidth, 13>{});
        case 14: return fn(std::integral_constant<BitWidth, 14>{});
        case 15: return fn(std::integral_constant<BitWidth, 15>{});
        case 16: return fn(std::integral_constant<BitWidth, 16>{});
        default: unreachable();
    }
}

} // namespace onpair
