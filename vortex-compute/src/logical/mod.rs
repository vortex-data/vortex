// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Logical boolean functions.

mod and;
mod and_kleene;
mod and_not;
mod not;
mod or;
mod or_kleene;

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

    /// Perform a logical AND operation between two values.
    fn and_kleene(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical AND NOT operations.
pub trait LogicalAndNot<Rhs = Self> {
    /// The resulting type after performing the logical AND NOT operation.
    type Output;

    /// Perform a logical AND operation between two values.
    fn and_not(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical OR operations.
pub trait LogicalOr<Rhs = Self> {
    /// The resulting type after performing the logical AND operation.
    type Output;

    /// Perform a logical OR operation between two values.
    fn or(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical OR KLEENE operations.
pub trait LogicalOrKleene<Rhs = Self> {
    /// The resulting type after performing the logical AND operation.
    type Output;

    /// Perform a logical OR KLEENE operation between two values.
    fn or_kleene(self, other: Rhs) -> Self::Output;
}

/// Trait for performing logical NOT operations.
pub trait LogicalNot {
    /// The resulting type after performing the logical AND NOT operation.
    type Output;

    /// Perform a logical NOT operation.
    fn not(self) -> Self::Output;
}
