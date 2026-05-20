// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::child_to_validity;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolDataParts;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::primitive::NativeValue;
use crate::builtins::ArrayBuiltins;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;

impl CompareKernel for Patched {
    #[expect(
        clippy::cognitive_complexity,
        reason = "complexity is from nested match_each_* macros"
    )]
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // We only accelerate comparisons for primitives
        if !lhs.dtype().is_primitive() {
            return Ok(None);
        }

        // We only accelerate comparisons against constants
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        // NOTE: due to offset, it's possible that the inner.len != array.len.
        //  We slice the inner before performing the comparison.
        let result = lhs
            .inner()
            .binary(
                ConstantArray::new(constant.clone(), lhs.len()).into_array(),
                operator.into(),
            )?
            .execute::<Canonical>(ctx)?
            .into_bool();

        let validity = child_to_validity(result.slots()[0].as_ref(), result.dtype().nullability());
        let len = result.len();
        let BoolDataParts { bits, offset, len } = result.into_data().into_parts(len);

        let mut bits = BitBufferMut::from_buffer(bits.unwrap_host().into_mut(), offset, len);

        let indices = lhs.patch_indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = lhs.patch_values().clone().execute::<PrimitiveArray>(ctx)?;
        let indices_ptype = indices.ptype();

        match_each_native_ptype!(values.ptype(), |V| {
            let offset = lhs.offset();
            let values = values.as_slice::<V>();
            let constant = constant
                .as_primitive()
                .as_::<V>()
                .vortex_expect("compare constant not null");

            match_each_unsigned_integer_ptype!(indices_ptype, |I| {
                let apply_patches = ApplyPatches {
                    bits: &mut bits,
                    offset,
                    indices: indices.as_slice::<I>(),
                    values,
                    constant,
                };

                match operator {
                    CompareOperator::Eq => {
                        apply_patches.apply(|l, r| NativeValue(l) == NativeValue(r));
                    }
                    CompareOperator::NotEq => {
                        apply_patches.apply(|l, r| NativeValue(l) != NativeValue(r));
                    }
                    CompareOperator::Gt => {
                        apply_patches.apply(|l, r| NativeValue(l) > NativeValue(r));
                    }
                    CompareOperator::Gte => {
                        apply_patches.apply(|l, r| NativeValue(l) >= NativeValue(r));
                    }
                    CompareOperator::Lt => {
                        apply_patches.apply(|l, r| NativeValue(l) < NativeValue(r));
                    }
                    CompareOperator::Lte => {
                        apply_patches.apply(|l, r| NativeValue(l) <= NativeValue(r));
                    }
                }
            });
        });

        let result = BoolArray::new(bits.freeze(), validity);
        Ok(Some(result.into_array()))
    }
}

struct ApplyPatches<'a, I: IntegerPType, V: NativePType> {
    bits: &'a mut BitBufferMut,
    offset: usize,
    indices: &'a [I],
    values: &'a [V],
    constant: V,
}

impl<I: IntegerPType, V: NativePType> ApplyPatches<'_, I, V> {
    fn apply<F>(self, cmp: F)
    where
        F: Fn(V, V) -> bool,
    {
        for (&patch_index, &patch_value) in std::iter::zip(self.indices, self.values) {
            let bit_index: usize = patch_index.as_();
            // Skip any indices < the offset.
            if bit_index < self.offset {
                continue;
            }
            let bit_index = bit_index - self.offset;
            if bit_index >= self.bits.len() {
                continue;
            }
            if cmp(patch_value, self.constant) {
                self.bits.set(bit_index)
            } else {
                self.bits.unset(bit_index)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;
    use crate::scalar_fn::fns::binary::CompareKernel;
    use crate::scalar_fn::fns::operators::CompareOperator;
    use crate::validity::Validity;

    #[test]
    fn test_basic() {
        let lhs = PrimitiveArray::from_iter(0u32..512).into_array();
        let patches = Patches::new(
            512,
            0,
            buffer![509u16, 510, 511].into_array(),
            buffer![u32::MAX; 3].into_array(),
            None,
        )
        .unwrap();

        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)
            .unwrap()
            .into_array()
            .try_downcast::<Patched>()
            .unwrap();

        let rhs = ConstantArray::new(u32::MAX, 512).into_array();

        let result =
            <Patched as CompareKernel>::compare(lhs.as_view(), &rhs, CompareOperator::Eq, &mut ctx)
                .unwrap()
                .unwrap();

        let expected =
            BoolArray::from_indices(512, [509, 510, 511], Validity::NonNullable).into_array();

        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_with_offset() {
        let lhs = PrimitiveArray::from_iter(0u32..512).into_array();
        let patches = Patches::new(
            512,
            0,
            buffer![5u16, 510, 511].into_array(),
            buffer![u32::MAX; 3].into_array(),
            None,
        )
        .unwrap();

        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx).unwrap();
        // Slice the array so that the first patch should be skipped.
        let lhs_ref = lhs.into_array().slice(10..512).unwrap().optimize().unwrap();
        let lhs = lhs_ref.try_downcast::<Patched>().unwrap();

        assert_eq!(lhs.len(), 502);

        let rhs = ConstantArray::new(u32::MAX, lhs.len()).into_array();

        let result =
            <Patched as CompareKernel>::compare(lhs.as_view(), &rhs, CompareOperator::Eq, &mut ctx)
                .unwrap()
                .unwrap();

        let expected = BoolArray::from_indices(502, [500, 501], Validity::NonNullable).into_array();

        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_subnormal_f32() -> VortexResult<()> {
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
        )?;

        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());
        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)?
            .into_array()
            .try_downcast::<Patched>()
            .map_err(|_| vortex_err!("expected patched array"))?;

        let rhs = ConstantArray::new(subnormal, 512).into_array();

        let result = <Patched as CompareKernel>::compare(
            lhs.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .ok_or_else(|| vortex_err!("expected compare result"))?;

        let expected = BoolArray::from_indices(512, [510], Validity::NonNullable).into_array();

        assert_arrays_eq!(expected, result);
        Ok(())
    }

    #[test]
    fn test_pos_neg_zero() -> VortexResult<()> {
        let lhs = PrimitiveArray::from_iter([-0.0f32; 10]).into_array();

        let patches = Patches::new(
            10,
            0,
            buffer![5u16, 6, 7, 8, 9].into_array(),
            buffer![f32::NAN, f32::NEG_INFINITY, 0f32, -0.0f32, f32::INFINITY].into_array(),
            None,
        )?;

        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());
        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)?
            .into_array()
            .try_downcast::<Patched>()
            .map_err(|_| vortex_err!("expected patched array"))?;

        let rhs = ConstantArray::new(0.0f32, 10).into_array();

        let result = <Patched as CompareKernel>::compare(
            lhs.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .ok_or_else(|| vortex_err!("expected compare result"))?;

        let expected = BoolArray::from_indices(10, [7], Validity::NonNullable).into_array();

        assert_arrays_eq!(expected, result);

        Ok(())
    }
}
