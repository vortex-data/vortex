// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex-native primitive vs constant comparison.
//!
//! Replaces the Arrow round-trip for `Primitive cmp Constant`. The hot loop chunks 8 input
//! elements into one output byte so the compiler can vectorize cleanly (~3× faster than the
//! `collect_bool` path Vortex was using through Arrow).

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::arrays::primitive::compute::chunked_pack::chunked_pack;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

impl CompareKernel for Primitive {
    fn compare(
        lhs: ArrayView<'_, Primitive>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle vs-constant. Vec-vec falls through to Arrow.
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };
        if rhs_const.is_null() {
            // Already handled upstream (execute_compare short-circuits), but be safe.
            return Ok(None);
        }
        let ptype = lhs.ptype();
        let rhs_ptype = match rhs_const.dtype() {
            DType::Primitive(p, _) => *p,
            _ => return Ok(None),
        };
        if ptype != rhs_ptype {
            // Mixed-ptype compare; let Arrow handle the cast + cmp.
            return Ok(None);
        }

        let bits: BitBuffer = match_each_native_ptype!(ptype, |T| {
            let Ok(needle) = T::try_from(&rhs_const) else {
                return Ok(None);
            };
            cmp_chunked::<T>(lhs.as_slice::<T>(), needle, operator)
        });

        let result_nullable = lhs.dtype().is_nullable() || rhs_const.dtype().is_nullable();
        let validity = if !result_nullable {
            Validity::NonNullable
        } else {
            match lhs.validity()? {
                Validity::NonNullable => Validity::AllValid,
                v => v,
            }
        };
        Ok(Some(BoolArray::new(bits, validity).into_array()))
    }
}

/// `cmp(slice[i], needle, op)` packed into a `BitBuffer`. Dispatches the comparison
/// operator once at the top so the inner loop is monomorphic and vectorizes.
#[inline]
fn cmp_chunked<T: NativePType>(slice: &[T], needle: T, op: CompareOperator) -> BitBuffer {
    match op {
        CompareOperator::Lt => chunked_pack(slice, |v| v.is_lt(needle)),
        CompareOperator::Lte => chunked_pack(slice, |v| v.is_le(needle)),
        CompareOperator::Gt => chunked_pack(slice, |v| v.is_gt(needle)),
        CompareOperator::Gte => chunked_pack(slice, |v| v.is_ge(needle)),
        CompareOperator::Eq => chunked_pack(slice, |v| v.is_eq(needle)),
        CompareOperator::NotEq => chunked_pack(slice, |v| !v.is_eq(needle)),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::Canonical;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar_fn::fns::operators::Operator;

    fn run(
        arr: PrimitiveArray,
        scalar: ArrayRef,
        op: Operator,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        arr.into_array()
            .binary(scalar, op)?
            .execute::<Canonical>(&mut ctx)
            .map(Into::into)
    }

    #[test]
    fn cmp_lt_u16() -> VortexResult<()> {
        let arr: PrimitiveArray = (0..1000u16).collect();
        let scalar = ConstantArray::new(500u16, 1000).into_array();
        let r = run(arr, scalar, Operator::Lt)?;
        // Sanity: first 500 true, rest false.
        let bool_arr = r.as_::<crate::arrays::Bool>().clone();
        let bits = bool_arr.to_bit_buffer();
        assert_eq!(bits.true_count(), 500);
        assert!(bits.value(0));
        assert!(bits.value(499));
        assert!(!bits.value(500));
        assert!(!bits.value(999));
        Ok(())
    }

    #[test]
    fn cmp_eq_i64() -> VortexResult<()> {
        let arr = buffer![10i64, 20, 30, 20, 10, 40];
        let arr = PrimitiveArray::new(arr, Validity::NonNullable);
        let scalar = ConstantArray::new(20i64, 6).into_array();
        let r = run(arr, scalar, Operator::Eq)?;
        let bool_arr = r.as_::<crate::arrays::Bool>().clone();
        let bits = bool_arr.to_bit_buffer();
        assert_eq!(
            (0..6).map(|i| bits.value(i)).collect::<Vec<_>>(),
            vec![false, true, false, true, false, false]
        );
        Ok(())
    }

    #[test]
    fn cmp_gte_tail() -> VortexResult<()> {
        // Length not a multiple of 8 exercises tail handling.
        let arr: PrimitiveArray = (0..13u32).collect();
        let scalar = ConstantArray::new(10u32, 13).into_array();
        let r = run(arr, scalar, Operator::Gte)?;
        let bool_arr = r.as_::<crate::arrays::Bool>().clone();
        let bits = bool_arr.to_bit_buffer();
        for i in 0..13 {
            assert_eq!(bits.value(i), i >= 10, "i={i}");
        }
        Ok(())
    }

    #[test]
    fn cmp_f32_nan_handled() -> VortexResult<()> {
        // NaN compared with anything via standard ops returns false, except NotEq.
        let arr =
            PrimitiveArray::new(buffer![1.0f32, f32::NAN, 2.0], Validity::NonNullable);
        let scalar = ConstantArray::new(1.5f32, 3).into_array();
        let r = run(arr, scalar, Operator::Lt)?;
        let bool_arr = r.as_::<crate::arrays::Bool>().clone();
        let bits = bool_arr.to_bit_buffer();
        // 1.0 < 1.5 -> true; NaN < 1.5 -> false; 2.0 < 1.5 -> false
        assert_eq!(
            (0..3).map(|i| bits.value(i)).collect::<Vec<_>>(),
            vec![true, false, false]
        );
        Ok(())
    }

    #[test]
    fn cmp_with_nullable_lhs() -> VortexResult<()> {
        let arr =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(2), None]);
        let scalar = ConstantArray::new(2i32, 5).into_array();
        let r = run(arr, scalar, Operator::Lt)?;
        // Bits at null positions are unspecified; check non-null ones via canonical.
        let bool_arr = r.as_::<crate::arrays::Bool>().clone();
        let bits = bool_arr.to_bit_buffer();
        // Expected non-null bits: 1<2=true, 3<2=false, 2<2=false
        assert!(bits.value(0));
        assert!(!bits.value(2));
        assert!(!bits.value(3));
        Ok(())
    }
}
