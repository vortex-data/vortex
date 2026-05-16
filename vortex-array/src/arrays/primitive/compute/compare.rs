// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex-native primitive vs constant comparison.
//!
//! Replaces the Arrow round-trip for `Primitive cmp Constant`. The hot loop chunks 8 input
//! elements into one output byte so the compiler can vectorize cleanly (~3× faster than the
//! `collect_bool` path Vortex was using through Arrow).

use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

impl CompareKernel for Primitive {
    fn compare(
        lhs: ArrayView<'_, Primitive>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
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
            let lhs_validity = lhs.validity()?;
            match lhs_validity {
                Validity::NonNullable => Validity::AllValid,
                v => v,
            }
        };
        let _ = ctx;
        let _ = PType::U8;
        Ok(Some(BoolArray::new(bits, validity).into_array()))
    }
}

/// Pack 8 cmps into a byte at a time so the compiler can vectorize the body. The naive
/// `collect_bool` path Vortex uses via Arrow's `cmp::lt` was ~3× slower at every size past
/// L1 (see `cpu_take_vs_cmp` bench).
#[inline]
fn cmp_chunked<T: NativePType>(slice: &[T], needle: T, op: CompareOperator) -> BitBuffer {
    let len = slice.len();
    let bytes_len = len.div_ceil(8);
    let mut bytes = ByteBufferMut::zeroed(bytes_len);

    match op {
        CompareOperator::Lt => pack_chunks(slice, needle, len, &mut bytes, |a, b| a.is_lt(b)),
        CompareOperator::Lte => pack_chunks(slice, needle, len, &mut bytes, |a, b| a.is_le(b)),
        CompareOperator::Gt => pack_chunks(slice, needle, len, &mut bytes, |a, b| a.is_gt(b)),
        CompareOperator::Gte => pack_chunks(slice, needle, len, &mut bytes, |a, b| a.is_ge(b)),
        CompareOperator::Eq => pack_chunks(slice, needle, len, &mut bytes, |a, b| a.is_eq(b)),
        CompareOperator::NotEq => pack_chunks(slice, needle, len, &mut bytes, |a, b| !a.is_eq(b)),
    }

    // Build a BitBuffer over exactly `len` bits.
    BitBuffer::new(bytes.freeze(), len)
}

#[inline(always)]
fn pack_chunks<T: NativePType, F: Fn(T, T) -> bool>(
    slice: &[T],
    needle: T,
    len: usize,
    bytes: &mut ByteBufferMut,
    pred: F,
) {
    let full = len / 8;
    let dst = bytes.as_mut_slice();
    for chunk_idx in 0..full {
        let base = chunk_idx * 8;
        let mut b = 0u8;
        // The inner loop is fully unrolled and vectorizes for primitive cmps.
        for j in 0..8 {
            // SAFETY: base + j < full*8 <= len.
            let v = unsafe { *slice.get_unchecked(base + j) };
            b |= u8::from(pred(v, needle)) << j;
        }
        // SAFETY: chunk_idx < full <= bytes_len.
        unsafe { *dst.get_unchecked_mut(chunk_idx) = b };
    }
    let tail = full * 8;
    if tail < len {
        let mut b = 0u8;
        for j in 0..(len - tail) {
            // SAFETY: tail + j < len.
            let v = unsafe { *slice.get_unchecked(tail + j) };
            b |= u8::from(pred(v, needle)) << j;
        }
        // SAFETY: full < bytes_len when len % 8 != 0.
        unsafe { *dst.get_unchecked_mut(full) = b };
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
