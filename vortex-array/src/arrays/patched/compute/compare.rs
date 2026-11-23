// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comparisons of patched arrays against constants.
//!
//! Comparisons against a constant compare the inner array, with its own kernel, and the patch
//! values, and patch the compared values over the inner result in one step. Over a constant
//! inner array, as in a sparse array, the fill value is compared directly.

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PATCH_CHUNK_SIZE;
use crate::arrays::patched::PatchLayout;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::execute::PatchedCanonicalParts;
use crate::arrays::patched::execute::PatchedInner;
use crate::arrays::patched::execute::execute_canonical;
use crate::arrays::patched::execute::to_canonical;
use crate::arrays::patched::patch_validity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::binary::CompareKernel;
use crate::scalar_fn::builtin::binary::scalar_cmp;
use crate::scalar_fn::builtin::operators::CompareOperator;
use crate::scalar_fn::builtin::operators::Operator;
use crate::validity::Validity;

impl CompareKernel for Patched {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs) = rhs.as_constant() else {
            return Ok(None);
        };

        if !lhs.inner().is::<Constant>()
            && let Some(result) = compare_primitive_fused(lhs, rhs, operator, ctx)?
        {
            return Ok(Some(result));
        }

        let values = compare_values(lhs.patch_values(), rhs, operator, ctx)?;
        let dtype = values.dtype().clone();

        // Over a constant inner array, compare the fill value directly.
        if let Some(fill) = lhs.inner().as_opt::<Constant>() {
            let fill = scalar_cmp(fill.scalar(), rhs, operator)?.cast(&dtype)?;
            let inner = ConstantArray::new(fill, lhs.len()).into_array();
            // SAFETY: the inner array and the compared values keep their lengths, and share the
            // dtype of the comparison.
            let patched = unsafe { Patched::with_inner_and_values(&lhs, dtype, inner, values) };
            return Ok(Some(patched.into_array()));
        }

        // Otherwise compare the inner array, which uses its own comparison kernel, such as a
        // bit-packed one, and patch the compared values over the result it owns.
        let inner = lhs
            .inner()
            .binary(
                ConstantArray::new(rhs.clone(), lhs.len()).into_array(),
                Operator::from(operator),
            )?
            .execute::<Canonical>(ctx)?;
        let values = to_canonical(values, ctx)?;
        let layout = lhs.execute_layout(ctx)?;
        let layout = layout.layout()?;
        let result = execute_canonical(
            PatchedCanonicalParts {
                dtype,
                layout,
                lane_offsets: lhs.lane_offsets().clone(),
                patch_indices: lhs.patch_indices().clone(),
                inner: PatchedInner::Canonical(inner),
                values,
            },
            ctx,
        )?;
        Ok(Some(result))
    }
}

/// Compare a patched array of sparse, non-null primitive patch values: compare the inner array with its
/// own kernel, then write the comparison of each patch value straight over its bit, as an
/// encoding with interior patches does, without materializing the compared patch values.
fn compare_primitive_fused(
    lhs: ArrayView<'_, Patched>,
    rhs: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let values = lhs.patch_values();
    let DType::Primitive(ptype, _) = values.dtype() else {
        return Ok(None);
    };
    let ptype = *ptype;
    if rhs.is_null()
        || !rhs.dtype().is_primitive()
        || rhs.dtype().as_ptype() != ptype
        || !values.validity()?.definitely_no_nulls()
    {
        return Ok(None);
    }

    // Dense patches, averaging more than 32 per chunk, are faster through the word-at-a-time bit
    // patching of the general path.
    let layout = lhs.execute_layout(ctx)?;
    let layout = layout.layout()?;
    if layout.all_patches().len() > 32 * layout.n_chunks() {
        return Ok(None);
    }

    let inner = lhs
        .inner()
        .binary(
            ConstantArray::new(rhs.clone(), lhs.len()).into_array(),
            Operator::from(operator),
        )?
        .execute::<BoolArray>(ctx)?;
    let values = values.clone().execute::<PrimitiveArray>(ctx)?;

    // The patched rows become valid, as the patch values are.
    let validity = patch_validity(
        inner.validity()?,
        Validity::AllValid,
        values.len(),
        &layout,
        ctx,
    )?;
    let mut bits = inner
        .into_bit_buffer()
        .try_into_mut()
        .unwrap_or_else(|bits| BitBufferMut::copy_from(&bits));
    match_each_native_ptype!(ptype, |T| {
        let values = values.as_slice::<T>();
        let rhs = rhs
            .as_primitive()
            .as_::<T>()
            .vortex_expect("a valid primitive scalar of the values' type");
        match operator {
            CompareOperator::Eq => patch_compared(&mut bits, &layout, values, |v| v.is_eq(rhs)),
            CompareOperator::NotEq => patch_compared(&mut bits, &layout, values, |v| !v.is_eq(rhs)),
            CompareOperator::Gt => patch_compared(&mut bits, &layout, values, |v| v.is_gt(rhs)),
            CompareOperator::Gte => patch_compared(&mut bits, &layout, values, |v| v.is_ge(rhs)),
            CompareOperator::Lt => patch_compared(&mut bits, &layout, values, |v| v.is_lt(rhs)),
            CompareOperator::Lte => patch_compared(&mut bits, &layout, values, |v| v.is_le(rhs)),
        }
    });
    Ok(Some(BoolArray::new(bits.freeze(), validity).into_array()))
}

/// Set the bit of each patched position to `predicate` of its patch value.
#[inline]
fn patch_compared<T: Copy>(
    bits: &mut BitBufferMut,
    layout: &PatchLayout<'_>,
    values: &[T],
    predicate: impl Fn(T) -> bool,
) {
    assert_eq!(bits.len(), layout.len());
    layout.for_each(|pos, patch| bits.set_to(pos, predicate(values[patch])));
}

/// Compare the patch values against `rhs`.
///
/// A few primitive patch values are compared directly, with the predicates of the primitive
/// comparison kernel, rather than through a separately executed comparison.
fn compare_values(
    values: &ArrayRef,
    rhs: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let DType::Primitive(ptype, nullability) = values.dtype() else {
        return values.binary(
            ConstantArray::new(rhs.clone(), values.len()).into_array(),
            Operator::from(operator),
        );
    };
    // Many values are faster through the vectorized comparison kernel.
    if values.len() > 4 * PATCH_CHUNK_SIZE
        || rhs.is_null()
        || !rhs.dtype().is_primitive()
        || rhs.dtype().as_ptype() != *ptype
    {
        return values.binary(
            ConstantArray::new(rhs.clone(), values.len()).into_array(),
            Operator::from(operator),
        );
    }
    let nullability = *nullability | rhs.dtype().nullability();
    let values = values.clone().execute::<PrimitiveArray>(ctx)?;
    let bits = match_each_native_ptype!(*ptype, |T| {
        let rhs = rhs
            .as_primitive()
            .as_::<T>()
            .vortex_expect("a valid primitive scalar of the values' type");
        compare_native(values.as_slice::<T>(), rhs, operator)
    });
    let validity = values.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits, validity).into_array())
}

fn compare_native<T: NativePType>(values: &[T], rhs: T, operator: CompareOperator) -> BitBuffer {
    #[inline]
    fn collect<T: NativePType>(values: &[T], predicate: impl Fn(T) -> bool) -> BitBuffer {
        // SAFETY: `collect_bool` calls the closure with indices below `values.len()`.
        BitBuffer::collect_bool(values.len(), |i| {
            predicate(unsafe { *values.get_unchecked(i) })
        })
    }
    match operator {
        CompareOperator::Eq => collect(values, |v| v.is_eq(rhs)),
        CompareOperator::NotEq => collect(values, |v| !v.is_eq(rhs)),
        CompareOperator::Gt => collect(values, |v| v.is_gt(rhs)),
        CompareOperator::Gte => collect(values, |v| v.is_ge(rhs)),
        CompareOperator::Lt => collect(values, |v| v.is_lt(rhs)),
        CompareOperator::Lte => collect(values, |v| v.is_le(rhs)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;
    use crate::scalar_fn::builtin::operators::Operator;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_basic() {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = PrimitiveArray::from_iter(0u32..512).into_array();
        let patches = Patches::new(
            512,
            0,
            buffer![509u16, 510, 511].into_array(),
            buffer![u32::MAX; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)
            .unwrap()
            .into_array()
            .try_downcast::<Patched>()
            .unwrap();

        let rhs = ConstantArray::new(u32::MAX, 512).into_array();

        let result = lhs.into_array().binary(rhs, Operator::Eq).unwrap();

        let expected =
            BoolArray::from_indices(512, [509, 510, 511], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_with_offset() {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = PrimitiveArray::from_iter(0u32..512).into_array();
        let patches = Patches::new(
            512,
            0,
            buffer![5u16, 510, 511].into_array(),
            buffer![u32::MAX; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx).unwrap();
        // Slice the array so that the first patch should be skipped.
        let lhs_ref = lhs
            .into_array()
            .slice(10..512)
            .unwrap()
            .optimize_ctx(&SESSION)
            .unwrap();
        let lhs = lhs_ref.try_downcast::<Patched>().unwrap();

        assert_eq!(lhs.len(), 502);

        let rhs = ConstantArray::new(u32::MAX, lhs.len()).into_array();

        let result = lhs.into_array().binary(rhs, Operator::Eq).unwrap();

        let expected = BoolArray::from_indices(502, [500, 501], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_subnormal_f32() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Subnormal f32 values are smaller than f32::MIN_POSITIVE but greater than 0
        let subnormal: f32 = f32::MIN_POSITIVE / 2.0;
        assert!(subnormal > 0.0 && subnormal < f32::MIN_POSITIVE);

        let lhs = PrimitiveArray::from_iter((0..512).map(|i| i as f32)).into_array();

        let patches = Patches::new(
            512,
            0,
            buffer![509u16, 510, 511].into_array(),
            buffer![f32::NAN, subnormal, f32::NEG_INFINITY].into_array(),
            None,
            &mut ctx,
        )?;

        let mut ctx = SESSION.create_execution_ctx();
        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)?
            .into_array()
            .try_downcast::<Patched>()
            .map_err(|_| vortex_err!("expected patched array"))?;

        let rhs = ConstantArray::new(subnormal, 512).into_array();

        let result = lhs.into_array().binary(rhs, Operator::Eq)?;

        let expected = BoolArray::from_indices(512, [510], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_pos_neg_zero() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = PrimitiveArray::from_iter([-0.0f32; 10]).into_array();

        let patches = Patches::new(
            10,
            0,
            buffer![5u16, 6, 7, 8, 9].into_array(),
            buffer![f32::NAN, f32::NEG_INFINITY, 0f32, -0.0f32, f32::INFINITY].into_array(),
            None,
            &mut ctx,
        )?;

        let mut ctx = SESSION.create_execution_ctx();
        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)?
            .into_array()
            .try_downcast::<Patched>()
            .map_err(|_| vortex_err!("expected patched array"))?;

        let rhs = ConstantArray::new(0.0f32, 10).into_array();

        let result = lhs.into_array().binary(rhs, Operator::Eq)?;

        let expected = BoolArray::from_indices(10, [7], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }
}
