#pragma once
#include <onpair/core/dictionary_view.h>
#include <onpair/core/store_view.h>
#include <onpair/decoding/decoder.h>
#include <onpair/search/automata/scan.h>
#include <onpair/search/automata/kmp_automaton.h>
#include <onpair/search/automata/prefix_automaton.h>
#include <onpair/search/eq_search.h>
#include <concepts>
#include <cstddef>
#include <functional>
#include <string_view>
#include <type_traits>
#include <vector>

namespace onpair {

// ─────────────────────────────────────────────────────────────────────────────
// DECOMPRESS_BUFFER_PADDING
// ─────────────────────────────────────────────────────────────────────────────
// Required extra bytes at the end of every decompress() output buffer.
// Equals MAX_TOKEN_SIZE: the decoder copies this many bytes per token
// regardless of the token's true size (over-copy optimisation).

inline constexpr size_t DECOMPRESS_BUFFER_PADDING = MAX_TOKEN_SIZE;

class OnPairColumn;  // forward declaration

// ─────────────────────────────────────────────────────────────────────────────
// OnPairColumnView
// ─────────────────────────────────────────────────────────────────────────────
// Non-owning view over a compressed column.  Provides decompression and
// search operations.  Lifetime is tied to the underlying OnPairColumn.

class OnPairColumnView {
public:
    /* implicit */ OnPairColumnView(const OnPairColumn& col) noexcept;

    OnPairColumnView(StoreView sv, DictionaryView dv) noexcept
        : sv_(sv), dv_(dv) {}

    // ── Metadata ──────────────────────────────────────────────────────────────
    size_t   num_strings() const noexcept { return sv_.num_strings(); }
    BitWidth bits()        const noexcept { return sv_.bits(); }
    size_t   bytes_used()  const noexcept {
        return sv_.bytes_used() + dv_.bytes_used();
    }

    // ── Random access ─────────────────────────────────────────────────────────
    size_t decompress(size_t idx, char* buf) const noexcept {
        return decoding::decompress(sv_, dv_, idx,
                                    reinterpret_cast<uint8_t*>(buf));
    }

    // ── Bulk decompression ───────────────────────────────────────────────────
    size_t decompress_all(char* buf) const noexcept {
        return decoding::decompress_all(sv_, dv_,
                                        reinterpret_cast<uint8_t*>(buf));
    }

    size_t decompress_all(char* buf, uint32_t* out_offsets) const noexcept {
        return decoding::decompress_all(sv_, dv_,
                                        reinterpret_cast<uint8_t*>(buf),
                                        out_offsets);
    }

    // ── Generic automaton scan ────────────────────────────────────────────────
    // Accepts both lvalue automata and temporaries returned by operator
    // overloads (!, &&, ||).

    template<typename A, std::invocable<size_t> F>
        requires search::TokenAutomaton<std::remove_reference_t<A>>
    void scan(A&& aut, F&& on_match) const {
        const auto* packed = sv_.packed_data();
        const auto* bounds = sv_.boundaries();
        const size_t n = sv_.num_strings();
        dispatch_bits(sv_.bits(), [&](auto bits) {
            search::detail::scan_impl<bits.value>(aut, packed, bounds, n, on_match);
        });
    }

    template<typename A>
        requires search::TokenAutomaton<std::remove_reference_t<A>>
    std::vector<size_t> scan(A&& aut) const {
        std::vector<size_t> result;
        scan(aut, [&](size_t idx) { result.push_back(idx); });
        return result;
    }

    // ── Substring search (KMP) ────────────────────────────────────────────────

    template<std::invocable<size_t> F>
    void contains(std::string_view pattern, F&& on_match) const {
        search::KmpAutomaton kmp(pattern, dv_);
        scan(kmp, std::forward<F>(on_match));
    }

    std::vector<size_t> contains(std::string_view pattern) const {
        std::vector<size_t> result;
        contains(pattern, [&](size_t idx) { result.push_back(idx); });
        return result;
    }

    // ── Prefix search ─────────────────────────────────────────────────────────

    template<std::invocable<size_t> F>
    void starts_with(std::string_view prefix, F&& on_match) const {
        search::PrefixAutomaton pa(prefix, dv_);
        scan(pa, std::forward<F>(on_match));
    }

    std::vector<size_t> starts_with(std::string_view prefix) const {
        std::vector<size_t> result;
        starts_with(prefix, [&](size_t idx) { result.push_back(idx); });
        return result;
    }
    
    // ── Exact-match search ─────────────────────────────────────────────────────

    template<std::invocable<size_t> F>
    void equals(std::string_view value, F&& on_match) const {
        search::EQSearch em(value, dv_);
        const auto* packed = sv_.packed_data();
        const auto* bounds = sv_.boundaries();
        const size_t n = sv_.num_strings();
        dispatch_bits(sv_.bits(), [&](auto bits) {
            em.template scan<bits.value>(packed, bounds, n, on_match);
        });
    }

    std::vector<size_t> equals(std::string_view value) const {
        std::vector<size_t> result;
        equals(value, [&](size_t idx) { result.push_back(idx); });
        return result;
    }

    // ── Internal accessors ────────────────────────────────────────────────────
    StoreView      store()      const noexcept { return sv_; }
    DictionaryView dictionary() const noexcept { return dv_; }

private:
    StoreView      sv_;
    DictionaryView dv_;
};

} // namespace onpair
