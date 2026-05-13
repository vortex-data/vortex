#pragma once
#include <onpair/core/types.h>
#include <concepts>
#include <type_traits>

namespace onpair::search {

// ─────────────────────────────────────────────────────────────────────────────
// TokenAutomaton concept
// ─────────────────────────────────────────────────────────────────────────────
// Any type that can be driven token-by-token over a compressed string to
// detect a match.  The scan loop consumes ALL tokens of each string and then
// reads is_accepted() once — the automaton must reflect the final verdict after
// the last token.

template<typename A>
concept TokenAutomaton = requires(A a, Token t) {
    { a.step(t)       } -> std::same_as<void>;
    { a.is_accepted() } -> std::convertible_to<bool>;
    { a.reset()       } -> std::same_as<void>;
};

// Optional refinement: automata that signal when further scanning cannot change
// the verdict.  When present, the scan loop breaks as soon as is_dead() returns
// true.  Substring automata (KmpAutomaton, AhoCorasickAutomaton) return
// is_dead() == true once a match is found; the result is final regardless of
// remaining tokens.

template<typename A>
concept DeadDetectable = TokenAutomaton<A> && requires(const A a) {
    { a.is_dead() } -> std::convertible_to<bool>;
};

// ─────────────────────────────────────────────────────────────────────────────
// Automaton combinators
// ─────────────────────────────────────────────────────────────────────────────
// Composable zero-cost wrappers that build new TokenAutomata from existing
// ones — enabling boolean algebra over compressed-domain search.
//
// All combinators satisfy TokenAutomaton.  DeadDetectable is satisfied
// conditionally when the wrapped automata satisfy it.

// ── NegatedAutomaton<A> ───────────────────────────────────────────────────────
// Inverts is_accepted().  is_dead() is forwarded unchanged.

template<TokenAutomaton A>
struct NegatedAutomaton {
    A& inner;
    explicit NegatedAutomaton(A& a) noexcept : inner(a) {}
    void step(Token t)       { inner.step(t); }
    bool is_accepted() const { return !inner.is_accepted(); }
    void reset()             { inner.reset(); }
    bool is_dead() const requires DeadDetectable<A> { return inner.is_dead(); }
};

// ── AndAutomaton<A, B> ────────────────────────────────────────────────────────
// Accepts when both A and B accept.  Drives both automata on every token.
// Early exit fires as soon as either component definitively rejects.
//
// Precondition: a and b must refer to distinct automaton objects.  Both are
// stepped on every token, so aliasing (a and b pointing to the same object)
// corrupts the automaton state and produces wrong results.

template<TokenAutomaton A, TokenAutomaton B>
struct AndAutomaton {
    A& a; B& b;
    AndAutomaton(A& a, B& b) noexcept : a(a), b(b) {}
    void step(Token t)       { a.step(t); b.step(t); }
    bool is_accepted() const { return a.is_accepted() && b.is_accepted(); }
    void reset()             { a.reset(); b.reset(); }
    bool is_dead() const requires (DeadDetectable<A> || DeadDetectable<B>) {
        if constexpr (DeadDetectable<A>)
            if (a.is_dead() && !a.is_accepted()) return true;
        if constexpr (DeadDetectable<B>)
            if (b.is_dead() && !b.is_accepted()) return true;
        return false;
    }
};

// ── OrAutomaton<A, B> ─────────────────────────────────────────────────────────
// Accepts when either A or B accepts.  Drives both automata on every token.
// Early exit fires as soon as either component definitively accepts.
//
// Precondition: a and b must refer to distinct automaton objects.  Both are
// stepped on every token, so aliasing (a and b pointing to the same object)
// corrupts the automaton state and produces wrong results.

template<TokenAutomaton A, TokenAutomaton B>
struct OrAutomaton {
    A& a; B& b;
    OrAutomaton(A& a, B& b) noexcept : a(a), b(b) {}
    void step(Token t)       { a.step(t); b.step(t); }
    bool is_accepted() const { return a.is_accepted() || b.is_accepted(); }
    void reset()             { a.reset(); b.reset(); }
    bool is_dead() const requires (DeadDetectable<A> || DeadDetectable<B>) {
        if constexpr (DeadDetectable<A>)
            if (a.is_dead() && a.is_accepted()) return true;
        if constexpr (DeadDetectable<B>)
            if (b.is_dead() && b.is_accepted()) return true;
        return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Operator overloads
// ─────────────────────────────────────────────────────────────────────────────
// Syntactic sugar for building combinator trees:
//   !a        → NegatedAutomaton
//   a && b    → AndAutomaton
//   a || b    → OrAutomaton

template<typename A>
    requires TokenAutomaton<std::remove_reference_t<A>>
auto operator!(A&& a) noexcept
    -> NegatedAutomaton<std::remove_reference_t<A>> {
    return NegatedAutomaton<std::remove_reference_t<A>>(a);
}

template<typename A, typename B>
    requires TokenAutomaton<std::remove_reference_t<A>>
          && TokenAutomaton<std::remove_reference_t<B>>
auto operator&&(A&& a, B&& b) noexcept
    -> AndAutomaton<std::remove_reference_t<A>, std::remove_reference_t<B>> {
    return {a, b};
}

template<typename A, typename B>
    requires TokenAutomaton<std::remove_reference_t<A>>
          && TokenAutomaton<std::remove_reference_t<B>>
auto operator||(A&& a, B&& b) noexcept
    -> OrAutomaton<std::remove_reference_t<A>, std::remove_reference_t<B>> {
    return {a, b};
}

} // namespace onpair::search
