// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Simple binary logical operations: AND, OR, AND NOT.
//!
//! These operations apply a bitwise operation to the bits and AND the validity masks together
//! (null propagates). For Kleene three-valued logic, see the [`kleene`](super::kleene) module.

use std::ops::BitAnd;
use std::ops::BitOr;

use vortex_buffer::BitBuffer;

/// Marker type for the AND operation.
pub struct And;

/// Marker type for the OR operation.
pub struct Or;

/// Marker type for the AND NOT operation.
pub struct AndNot;

/// Trait for simple logical binary operations.
///
/// These operations apply a bitwise operation to the bits and AND the validity masks together.
pub trait LogicalBinaryOp {
    /// Apply the operation to two [`BitBuffer`]s.
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer;

    /// Apply the operation to two scalar boolean values.
    fn scalar_op(lhs: bool, rhs: bool) -> bool;
}

impl LogicalBinaryOp for And {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitand(rhs)
    }

    fn scalar_op(lhs: bool, rhs: bool) -> bool {
        lhs && rhs
    }
}

impl LogicalBinaryOp for Or {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitor(rhs)
    }

    fn scalar_op(lhs: bool, rhs: bool) -> bool {
        lhs || rhs
    }
}

impl LogicalBinaryOp for AndNot {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitand_not(rhs)
    }

    fn scalar_op(lhs: bool, rhs: bool) -> bool {
        lhs && !rhs
    }
}
