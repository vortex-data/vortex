// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Logical boolean functions.
//!
//! This module provides logical operations for boolean scalars, vectors, and datums:
//!
//! - **Simple operations** (`binary`): AND, OR, AND NOT. These propagate nulls.
//! - **Kleene operations** (`kleene`): AND KLEENE, OR KLEENE. These use Kleene three-valued
//!   logic where `false AND null = false` and `true OR null = true`.
//! - **Unary operations** (`not`): NOT.

// TODO: We want to add these logical traits on the owned versions of the operands so that we can do
// in-place operatinos.

mod binary;
mod kleene;
mod not;

pub use binary::And;
pub use binary::AndNot;
pub use binary::LogicalBinaryOp;
pub use binary::Or;
pub use kleene::KleeneAnd;
pub use kleene::KleeneBinaryOp;
pub use kleene::KleeneOr;

/// `(Bool, Bool) -> Bool` compute function.
pub trait LogicalOp<Op, Rhs = Self> {
    /// The resulting type after performing the logical operation.
    type Output;

    /// Perform a logical operation between two values.
    fn op(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical AND operations.
pub trait LogicalAnd<Rhs = Self> {
    /// The resulting type after performing the logical AND operation.
    type Output;

    /// Perform a logical AND operation between two values.
    fn and(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical AND KLEENE operations.
pub trait LogicalAndKleene<Rhs = Self> {
    /// The resulting type after performing the logical AND KLEENE operation.
    type Output;

    /// Perform a logical AND KLEENE operation between two values.
    fn and_kleene(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical AND NOT operations.
pub trait LogicalAndNot<Rhs = Self> {
    /// The resulting type after performing the logical AND NOT operation.
    type Output;

    /// Perform a logical AND NOT operation between two values.
    fn and_not(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical OR operations.
pub trait LogicalOr<Rhs = Self> {
    /// The resulting type after performing the logical OR operation.
    type Output;

    /// Perform a logical OR operation between two values.
    fn or(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical OR KLEENE operations.
pub trait LogicalOrKleene<Rhs = Self> {
    /// The resulting type after performing the logical OR KLEENE operation.
    type Output;

    /// Perform a logical OR KLEENE operation between two values.
    fn or_kleene(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical NOT operations.
pub trait LogicalNot {
    /// The resulting type after performing the logical NOT operation.
    type Output;

    /// Perform a logical NOT operation.
    fn not(self) -> Self::Output;
}
