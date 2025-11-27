// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::Scalar;

/// A trait for describing the signature of a scalar function including its properties.
pub trait Signature {
    /// Returns the arity (number of arguments) for this function.
    fn arity(&self) -> usize;

    /// Returns the display name of the nth argument for this function.
    fn name(&self, arg_idx: usize) -> Option<String>;

    /// The identity element `e` where `f(e, x) = f(x, e) = x`.
    ///
    /// When an argument is the identity element, the function can be
    /// eliminated entirely, returning the other argument unchanged.
    ///
    /// # Examples
    /// - `AND`: `true` (AND(true, x) → x)
    /// - `OR`: `false` (OR(false, x) → x)
    /// - `+`: `0` (0 + x → x)
    /// - `*`: `1` (1 * x → x)
    /// - `COALESCE`: `NULL` (COALESCE(NULL, x) → x)
    fn identity_element(&self) -> Option<Scalar> {
        None
    }

    /// The absorbing element `a` where `f(a, x) = f(x, a) = a`.
    ///
    /// When any argument is the absorbing element, the function short-circuits
    /// immediately, returning that element without evaluating other arguments.
    /// Also known as the "annihilator" or "zero element".
    ///
    /// # Examples
    /// - `AND`: `false` (AND(false, x) → false)
    /// - `OR`: `true` (OR(true, x) → true)
    /// - `*`: `0` (0 * x → 0)
    fn absorbing_element(&self) -> Option<Scalar> {
        None
    }

    /// Per-argument monotonicity of the function with respect to the
    /// natural ordering of the argument type.
    ///
    /// A function is isotone (order-preserving) in an argument if increasing
    /// that argument never decreases the result. It is antitone (order-reversing)
    /// if increasing the argument never increases the result.
    ///
    /// Monotonicity enables zone map / min-max index falsification: if we know
    /// `x ∈ [min, max]`, we can bound `f(x)` and potentially skip data.
    ///
    /// # Examples
    /// - `x < y`: antitone in x (larger x → less likely to be less than y),
    ///            isotone in y (larger y → more likely that x < y)
    /// - `x + y`: isotone in both arguments
    /// - `x - y`: isotone in x, antitone in y
    /// - `ABS(x)`: neither (non-monotonic)
    fn monotonicity(&self, arg_idx: usize) -> Monotonicity {
        _ = arg_idx;
        Monotonicity::default()
    }

    /// Whether argument order is irrelevant: `f(a, b) = f(b, a)`.
    ///
    /// Enables expression normalization (e.g., sorting arguments by column id)
    /// for better common subexpression elimination and pattern matching.
    ///
    /// # Examples
    /// - Commutative: `+`, `*`, `AND`, `OR`, `=`, `!=`, `MIN`, `MAX`
    /// - Non-commutative: `-`, `/`, `<`, `>`, `CONCAT`
    fn is_commutative(&self) -> bool {
        false
    }

    /// Whether `f(x, x) = x`.
    ///
    /// Enables simplification when the same expression appears multiple times
    /// as arguments to the function.
    ///
    /// # Examples
    /// - Idempotent: `AND`, `OR`, `MIN`, `MAX`
    /// - Non-idempotent: `+` (x + x = 2x), `*` (x * x = x²)
    fn is_idempotent(&self) -> bool {
        false
    }

    /// Whether `f(f(x)) = x` for unary functions.
    ///
    /// Enables cancellation of nested self-applications.
    ///
    /// # Examples
    /// - Involutions: `NOT`, `NEG` (for signed types), `REVERSE`
    /// - Non-involutions: `ABS`, `UPPER`, `LOWER`
    fn is_involution(&self) -> bool {
        false
    }

    /// How the function behaves when one or more arguments are NULL.
    ///
    /// Most functions propagate NULL (any NULL argument produces NULL output).
    /// Some functions have special NULL handling that can short-circuit
    /// evaluation or treat NULL as a meaningful value.
    ///
    /// Required for correct NULL semantics; may also enable optimizations
    /// when argument nullability is known from schema or statistics.
    fn null_handling(&self) -> NullHandling {
        NullHandling::default()
    }
}

/// Monotonicity of a function with respect to one of its arguments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Monotonicity {
    /// Order-preserving: `x ≤ y` implies `f(x) ≤ f(y)`.
    ///
    /// For zone map falsification, an isotone argument means we use the
    /// minimum bound to establish a lower bound on the result.
    Isotone,

    /// Order-reversing: `x ≤ y` implies `f(x) ≥ f(y)`.
    ///
    /// For zone map falsification, an antitone argument means we use the
    /// maximum bound to establish a lower bound on the result.
    Antitone,

    /// No monotonic relationship exists, or it is unknown.
    ///
    /// Zone map falsification cannot use this argument for pruning.
    #[default]
    None,
}

/// How a function handles NULL arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum NullHandling {
    /// NULL in any argument produces NULL output.
    ///
    /// This is standard SQL behavior for most scalar functions.
    /// Enables simplification when any argument is known to be NULL.
    Propagate,

    /// NULL is short-circuited when paired with the absorbing element.
    ///
    /// This is a special case where the absorbing element "wins" over NULL.
    ///
    /// # Examples
    /// - `AND_KLEENE(false, NULL)` → `false` (false absorbs NULL)
    /// - `OR_KLEENE(true, NULL)` → `true` (true absorbs NULL)
    AbsorbsNull,

    /// The function has special NULL semantics that don't follow
    /// simple propagation rules.
    ///
    /// This prevents any simplifications based on NULL arguments.
    ///
    /// # Examples
    /// - `IS NULL`, `IS NOT NULL`: NULL → true/false
    /// - `COALESCE`: returns first non-NULL argument
    /// - `NULLIF`: conditionally produces NULL
    #[default]
    Custom,
}
