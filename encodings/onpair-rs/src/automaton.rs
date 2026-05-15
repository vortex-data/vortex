// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/token_automaton.h`.
//
// Token-level automata that consume the bit-packed token stream of an
// `onpair_lib::Column` directly. The scan loop in [`Column::scan`] resets
// the automaton at each row, feeds every token, and inspects
// `is_accepted()` once after the last token (or after `is_dead()` becomes
// true, whichever is first).
//
// Build composite predicates via [`and`], [`or`], [`not`]; the wrappers
// also implement [`TokenAutomaton`], so they nest. Every concrete
// automaton must implement `step` / `is_accepted` / `reset`; the default
// `is_dead` returns `false`, which is correct for any automaton that
// never finalises before the end of the row.

use crate::types::Token;

/// Token-by-token streaming predicate. Reset once per row, stepped on every
/// token, read for the final verdict.
pub trait TokenAutomaton {
    fn step(&mut self, t: Token);
    fn is_accepted(&self) -> bool;
    fn reset(&mut self);
    /// `true` once the verdict cannot change regardless of remaining
    /// tokens. The scan loop uses this to skip the rest of a row.
    fn is_dead(&self) -> bool {
        false
    }
}

impl<A: TokenAutomaton + ?Sized> TokenAutomaton for &mut A {
    #[inline]
    fn step(&mut self, t: Token) {
        (**self).step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        (**self).is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        (**self).reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        (**self).is_dead()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Combinators — Negated / And / Or.
// ─────────────────────────────────────────────────────────────────────────────

/// `!A` — flips `is_accepted`. `is_dead` is forwarded unchanged.
pub struct Negated<A>(pub A);

impl<A: TokenAutomaton> TokenAutomaton for Negated<A> {
    #[inline]
    fn step(&mut self, t: Token) {
        self.0.step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        !self.0.is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        self.0.is_dead()
    }
}

/// `A AND B` — both must accept. Both step on every token. Early-exits
/// when either inner becomes dead in a state that proves rejection.
pub struct And<A, B>(pub A, pub B);

impl<A: TokenAutomaton, B: TokenAutomaton> TokenAutomaton for And<A, B> {
    #[inline]
    fn step(&mut self, t: Token) {
        self.0.step(t);
        self.1.step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        self.0.is_accepted() && self.1.is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        (self.0.is_dead() && !self.0.is_accepted())
            || (self.1.is_dead() && !self.1.is_accepted())
    }
}

/// `A OR B` — either may accept. Both step on every token. Early-exits
/// when either inner becomes dead in a state that proves acceptance.
pub struct Or<A, B>(pub A, pub B);

impl<A: TokenAutomaton, B: TokenAutomaton> TokenAutomaton for Or<A, B> {
    #[inline]
    fn step(&mut self, t: Token) {
        self.0.step(t);
        self.1.step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        self.0.is_accepted() || self.1.is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        (self.0.is_dead() && self.0.is_accepted())
            || (self.1.is_dead() && self.1.is_accepted())
    }
}

/// `not(a)` constructs a [`Negated`] wrapper.
pub fn not<A: TokenAutomaton>(a: A) -> Negated<A> {
    Negated(a)
}

/// `and(a, b)` constructs an [`And`] wrapper.
pub fn and<A: TokenAutomaton, B: TokenAutomaton>(a: A, b: B) -> And<A, B> {
    And(a, b)
}

/// `or(a, b)` constructs an [`Or`] wrapper.
pub fn or<A: TokenAutomaton, B: TokenAutomaton>(a: A, b: B) -> Or<A, B> {
    Or(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny test automaton that accepts a fixed token-id once seen.
    struct AcceptsToken {
        target: Token,
        seen: bool,
    }
    impl AcceptsToken {
        fn new(t: Token) -> Self {
            Self { target: t, seen: false }
        }
    }
    impl TokenAutomaton for AcceptsToken {
        fn step(&mut self, t: Token) {
            if t == self.target {
                self.seen = true;
            }
        }
        fn is_accepted(&self) -> bool {
            self.seen
        }
        fn reset(&mut self) {
            self.seen = false;
        }
        fn is_dead(&self) -> bool {
            self.seen
        }
    }

    fn drive<A: TokenAutomaton>(mut a: A, tokens: &[Token]) -> bool {
        a.reset();
        for &t in tokens {
            a.step(t);
            if a.is_dead() {
                break;
            }
        }
        a.is_accepted()
    }

    #[test]
    fn accepts_token_basic() {
        assert!(drive(AcceptsToken::new(7), &[1, 2, 3, 7, 9]));
        assert!(!drive(AcceptsToken::new(7), &[1, 2, 3, 8]));
    }

    #[test]
    fn negation_inverts() {
        assert!(!drive(not(AcceptsToken::new(7)), &[1, 7, 2]));
        assert!(drive(not(AcceptsToken::new(7)), &[1, 8, 2]));
    }

    #[test]
    fn and_requires_both() {
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(and(a, b), &[1, 2, 3]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(!drive(and(a, b), &[1, 3]));
    }

    #[test]
    fn or_requires_either() {
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(or(a, b), &[1, 9]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(or(a, b), &[9, 2]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(!drive(or(a, b), &[3, 4, 5]));
    }

    #[test]
    fn nested_and_not() {
        // A AND NOT B
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(and(a, not(b)), &[1, 3]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(!drive(and(a, not(b)), &[1, 2, 3]));
    }

    #[test]
    fn references_implement_trait() {
        let mut a = AcceptsToken::new(7);
        let result = drive(&mut a, &[1, 7]);
        assert!(result);
        // Inner state remains accepting after.
        assert!(a.is_accepted());
    }

    #[test]
    fn or_dead_when_accepted() {
        // Once an Or component accepts and is dead, the combinator is dead.
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        let mut comb = or(a, b);
        comb.reset();
        comb.step(1);
        assert!(comb.is_dead());
        assert!(comb.is_accepted());
    }
}
