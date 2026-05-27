// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Element-wise checked integer addition over two primitive arrays.
//!
//! This is a worked example of building an arrow-compatible compute kernel on top of the
//! autovectorizable lane kernels in [`vortex_buffer::lane_ops_indexed`]. It mirrors the
//! semantics of `arrow_arith::numeric::add` for integers: validity is the union (logical AND)
//! of the two operands' validity, and overflow only faults at a **valid** lane (an overflow
//! whose value lives at a null lane is ignored).
//!
//! Unlike arrow's checked add, the inner add loop here is branch-free and autovectorizes: the
//! validity union is computed once as a bitmap and overflow is folded into an OR-reduced flag
//! ([`try_map_nullable`]), so the per-lane work is a SIMD add plus a SIMD overflow compare.

use std::mem::MaybeUninit;

use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::LaneZip;
use vortex_buffer::lane_ops_indexed::try_map_nullable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Compute the element-wise checked sum of two integer [`PrimitiveArray`]s.
///
/// Both arrays must have the same length and the same integer `PType`. The result validity is
/// the logical AND of the two operands' validity. Overflow at a valid lane returns an error;
/// overflow at a null lane is ignored (matching `arrow_arith::numeric::add`).
///
/// # Errors
///
/// Returns an error if the lengths differ, the ptypes differ or are non-integer, or a valid
/// lane overflows.
pub fn checked_add(
    lhs: &PrimitiveArray,
    rhs: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let len = lhs.len();
    if rhs.len() != len {
        vortex_bail!("checked_add length mismatch: {len} != {}", rhs.len());
    }
    let ptype = lhs.ptype();
    if ptype != rhs.ptype() {
        vortex_bail!("checked_add ptype mismatch: {ptype} != {}", rhs.ptype());
    }
    if !ptype.is_int() {
        vortex_bail!("checked_add requires an integer ptype, got {ptype}");
    }

    // Validity union (arrow's null-buffer union), computed once outside the arithmetic loop.
    let combined = &lhs.validity()?.execute_mask(len, ctx)?.to_bit_buffer()
        & &rhs.validity()?.execute_mask(len, ctx)?.to_bit_buffer();

    let nullable = lhs.nullability().is_nullable() || rhs.nullability().is_nullable();

    match_each_integer_ptype!(ptype, |T| {
        let mut out: Vec<MaybeUninit<T>> = Vec::with_capacity(len);
        // SAFETY: `try_map_nullable` writes every lane before it returns `Ok`.
        unsafe { out.set_len(len) };

        let result = try_map_nullable(
            LaneZip::new(lhs.as_slice::<T>(), rhs.as_slice::<T>()),
            &combined,
            out.as_mut_slice(),
            |(a, b)| a.checked_add(b),
        );
        if let Err(idx) = result {
            vortex_bail!("checked_add overflow at index {idx}");
        }

        // SAFETY: every lane was initialized since `try_map_nullable` returned `Ok`.
        let values: Vec<T> = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(out) };
        let validity = if nullable {
            Validity::from(combined)
        } else {
            Validity::NonNullable
        };
        Ok(PrimitiveArray::new(Buffer::from(values), validity))
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests use unwrap for brevity")]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::LEGACY_SESSION;
    use crate::assert_arrays_eq;
    use crate::executor::VortexSessionExecute;

    fn ctx() -> ExecutionCtx {
        LEGACY_SESSION.create_execution_ctx()
    }

    #[test]
    fn adds_non_null() {
        let lhs = PrimitiveArray::from_iter([1u32, 2, 3, 4]);
        let rhs = PrimitiveArray::from_iter([10u32, 20, 30, 40]);
        let out = checked_add(&lhs, &rhs, &mut ctx()).unwrap();
        assert_arrays_eq!(out, PrimitiveArray::from_iter([11u32, 22, 33, 44]));
    }

    #[test]
    fn adds_with_nulls_unions_validity() {
        let lhs = PrimitiveArray::from_option_iter([Some(1u32), None, Some(3), Some(4)]);
        let rhs = PrimitiveArray::from_option_iter([Some(10u32), Some(20), None, Some(40)]);
        let out = checked_add(&lhs, &rhs, &mut ctx()).unwrap();
        // Lanes 1 and 2 are null in the union; 0 and 3 are valid.
        assert_arrays_eq!(
            out,
            PrimitiveArray::from_option_iter([Some(11u32), None, None, Some(44)])
        );
    }

    #[test]
    fn overflow_at_valid_lane_errors() {
        let lhs = PrimitiveArray::from_iter([1u32, u32::MAX, 3]);
        let rhs = PrimitiveArray::from_iter([1u32, 1, 1]);
        let err = checked_add(&lhs, &rhs, &mut ctx()).unwrap_err();
        assert!(err.to_string().contains("overflow at index 1"), "{err}");
    }

    #[test]
    fn overflow_at_null_lane_is_ignored() {
        // Lane 1 holds an overflowing physical value but is null -> union is null -> not an
        // error (arrow parity).
        let validity = Validity::from(BitBuffer::from(vec![true, false, true]));
        let lhs = PrimitiveArray::new(buffer![1u32, u32::MAX, 3], validity);
        let rhs = PrimitiveArray::from_iter([1u32, 1, 1]);
        let out = checked_add(&lhs, &rhs, &mut ctx()).unwrap();
        assert_arrays_eq!(
            out,
            PrimitiveArray::from_option_iter([Some(2u32), None, Some(4)])
        );
    }

    #[test]
    fn length_mismatch_errors() {
        let lhs = PrimitiveArray::from_iter([1u32, 2, 3]);
        let rhs = PrimitiveArray::from_iter([1u32, 2]);
        assert!(checked_add(&lhs, &rhs, &mut ctx()).is_err());
    }

    #[test]
    fn ptype_mismatch_errors() {
        let lhs = PrimitiveArray::from_iter([1u32, 2, 3]);
        let rhs = PrimitiveArray::from_iter([1i64, 2, 3]);
        assert!(checked_add(&lhs, &rhs, &mut ctx()).is_err());
    }
}
