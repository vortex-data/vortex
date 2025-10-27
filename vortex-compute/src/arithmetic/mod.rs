// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic operations on buffers and vectors.

mod buffer;

/// Performs addition that returns None instead of wrapping around on overflow.
pub trait CheckedAdd<Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the operation.
    fn checked_add(self, other: Rhs) -> Option<Self::Output>;
}

/// Performs subtraction that returns None instead of wrapping around on underflow.
pub trait CheckedSub<Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the operation.
    fn checked_sub(self, other: Rhs) -> Option<Self::Output>;
}

/// Performs multiplication that returns None instead of wrapping around on underflow or overflow.
pub trait CheckedMul<Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the operation.
    fn checked_mul(self, other: Rhs) -> Option<Self::Output>;
}

/// Performs division that returns None instead of panicking on division by zero and instead of
/// wrapping around on underflow and overflow.
pub trait CheckedDiv<Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the operation.
    fn checked_div(self, other: Rhs) -> Option<Self::Output>;
}
