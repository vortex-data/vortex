// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kleene three-valued logical operations: AND KLEENE, OR KLEENE.
//!
//! These operations implement Kleene's three-valued logic (K3), also used by SQL:
//! - `FALSE AND NULL = FALSE` (false absorbs null)
//! - `TRUE OR NULL = TRUE` (true absorbs null)
//!
//! For simple null-propagating operations, see the [`binary`](super::binary) module.

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_buffer::BitBuffer;

/// Marker type for the Kleene AND operation.
pub struct KleeneAnd;

/// Marker type for the Kleene OR operation.
pub struct KleeneOr;

/// Trait for Kleene three-valued logical binary operations.
///
/// Absorbing values produce a valid result regardless of the other operand:
/// - For AND: `FALSE` absorbs nulls (`FALSE AND NULL = FALSE`)
/// - For OR: `TRUE` absorbs nulls (`TRUE OR NULL = TRUE`)
pub trait KleeneBinaryOp {
    /// Apply the operation to two [`BitBuffer`]s.
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer;

    /// Returns a mask of positions with absorbing values.
    ///
    /// - AND: `FALSE` absorbs, so return `bits.not()` (false positions).
    /// - OR: `TRUE` absorbs, so return `bits.clone()` (true positions).
    fn absorb_bits(bits: &BitBuffer) -> BitBuffer;

    /// Apply the operation to two scalar `Option<bool>` values with Kleene semantics.
    fn scalar_op(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool>;
}

impl KleeneBinaryOp for KleeneAnd {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitand(rhs)
    }

    fn absorb_bits(bits: &BitBuffer) -> BitBuffer {
        bits.not() // `false` absorbs nulls.
    }

    fn scalar_op(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
        match (lhs, rhs) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        }
    }
}

impl KleeneBinaryOp for KleeneOr {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitor(rhs)
    }

    fn absorb_bits(bits: &BitBuffer) -> BitBuffer {
        bits.clone() // `true` absorbs nulls.
    }

    fn scalar_op(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
        match (lhs, rhs) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        }
    }
}
