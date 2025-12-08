// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Logical boolean functions.

use vortex_vector::BoolDatum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;

mod and;
mod and_kleene;
mod and_not;
mod not;
mod or;
mod or_kleene;

/// kleene `and` op
pub struct KleeneAnd;
/// kleene `or` op
pub struct KleeneOr;

/// (Bool, Bool) -> Bool compute function.
pub trait LogicalOp<Op, Rhs = Self> {
    /// The resulting type after performing the logical AND KLEENE operation.
    type Output;

    /// Perform a logical AND operation between two values.
    fn op(self, other: Rhs) -> Self::Output;
}

impl<Op> LogicalOp<Op> for BoolDatum
where
    for<'a> &'a BoolVector: LogicalOp<Op, Output = BoolVector>,
    for<'a> &'a BoolScalar: LogicalOp<Op, Output = BoolScalar>,
{
    type Output = Self;

    fn op(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BoolDatum::Vector(lhs_vec), BoolDatum::Vector(rhs_vec)) => {
                BoolDatum::Vector(lhs_vec.op(&rhs_vec))
            }
            (BoolDatum::Scalar(lhs_sc), BoolDatum::Scalar(rhs_sc)) => {
                BoolDatum::Scalar(lhs_sc.and_kleene(&rhs_sc))
            }
            // TODO: remove repeat
            (BoolDatum::Scalar(lhs_sc), BoolDatum::Vector(rhs_vec)) => BoolDatum::Vector(
                lhs_sc
                    .repeat(rhs_vec.len())
                    .freeze()
                    .into_bool()
                    .op(&rhs_vec),
            ),
            (BoolDatum::Vector(lhs_vec), BoolDatum::Scalar(rhs_sc)) => {
                BoolDatum::Vector(lhs_vec.op(&rhs_sc.repeat(lhs_vec.len()).freeze().into_bool()))
            }
        }
    }
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
