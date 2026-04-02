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
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayParts;
use crate::arrays::primitive::NativeValue;
use crate::builtins::ArrayBuiltins;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;

impl CompareKernel for Patched {
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
            .base_array()
            .binary(
                ConstantArray::new(constant.clone(), lhs.len()).into_array(),
                operator.into(),
            )?
            .execute::<Canonical>(ctx)?
            .into_bool();

        let BoolArrayParts {
            bits,
            offset,
            len,
            validity,
        } = result.into_parts();

        let mut bits = BitBufferMut::from_buffer(bits.unwrap_host().into_mut(), offset, len);

        let lane_offsets = lhs.lane_offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let indices = lhs.patch_indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = lhs.patch_values().clone().execute::<PrimitiveArray>(ctx)?;
        let n_lanes = lhs.n_lanes;

        match_each_native_ptype!(values.ptype(), |V| {
            let offset = lhs.offset;
            let indices = indices.as_slice::<u16>();
            let values = values.as_slice::<V>();
            let constant = constant
                .as_primitive()
                .as_::<V>()
                .vortex_expect("compare constant not null");

            let apply_patches = ApplyPatches {
                bits: &mut bits,
                offset,
                n_lanes,
                lane_offsets: lane_offsets.as_slice::<u32>(),
                indices,
                values,
                constant,
            };

            match operator {
                CompareOperator::Eq => {
                    apply_patches.apply(|l, r| NativeValue(l) == NativeValue(r))?;
                }
                CompareOperator::NotEq => {
                    apply_patches.apply(|l, r| NativeValue(l) != NativeValue(r))?;
                }
                CompareOperator::Gt => {
                    apply_patches.apply(|l, r| NativeValue(l) > NativeValue(r))?;
                }
                CompareOperator::Gte => {
                    apply_patches.apply(|l, r| NativeValue(l) >= NativeValue(r))?;
                }
                CompareOperator::Lt => {
                    apply_patches.apply(|l, r| NativeValue(l) < NativeValue(r))?;
                }
                CompareOperator::Lte => {
                    apply_patches.apply(|l, r| NativeValue(l) <= NativeValue(r))?;
                }
            }
        });

        let result = BoolArray::new(bits.freeze(), validity);
        Ok(Some(result.into_array()))
    }
}

struct ApplyPatches<'a, V: NativePType> {
    bits: &'a mut BitBufferMut,
    offset: usize,
    n_lanes: usize,
    lane_offsets: &'a [u32],
    indices: &'a [u16],
    values: &'a [V],
    constant: V,
}

impl<V: NativePType> ApplyPatches<'_, V> {
    fn apply<F>(self, cmp: F) -> VortexResult<()>
    where
        F: Fn(V, V) -> bool,
    {
        for index in 0..(self.lane_offsets.len() - 1) {
            let chunk = index / self.n_lanes;

            let lane_start = self.lane_offsets[index] as usize;
            let lane_end = self.lane_offsets[index + 1] as usize;

            for (&patch_index, &patch_value) in std::iter::zip(
                &self.indices[lane_start..lane_end],
                &self.values[lane_start..lane_end],
            ) {
                let bit_index = chunk * 1024 + patch_index as usize;
                // Skip any indices < the offset.
                if bit_index < self.offset {
                    continue;
                }
                let bit_index = bit_index - self.offset;
                if bit_index >= self.bits.len() {
                    break;
                }
                if cmp(patch_value, self.constant) {
                    self.bits.set(bit_index)
                } else {
                    self.bits.unset(bit_index)
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::array::Array;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::Patched;
    use crate::arrays::PatchedArray;
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

        let lhs = Array::<Patched>::try_from_data(
            PatchedArray::from_array_and_patches(lhs, &patches, &mut ctx).unwrap(),
        )
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

        let lhs = PatchedArray::from_array_and_patches(lhs, &patches, &mut ctx).unwrap();
        // Slice the array so that the first patch should be skipped.
        let lhs_ref = lhs.into_array().slice(10..512).unwrap().optimize().unwrap();
        let lhs = lhs_ref.try_into::<Patched>().unwrap();

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
        let lhs = Array::<Patched>::try_from_data(PatchedArray::from_array_and_patches(
            lhs, &patches, &mut ctx,
        )?)?;

        let rhs = ConstantArray::new(subnormal, 512).into_array();

        let result = <Patched as CompareKernel>::compare(
            lhs.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .unwrap();

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
        let lhs = Array::<Patched>::try_from_data(PatchedArray::from_array_and_patches(
            lhs, &patches, &mut ctx,
        )?)?;

        let rhs = ConstantArray::new(0.0f32, 10).into_array();

        let result = <Patched as CompareKernel>::compare(
            lhs.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .unwrap();

        let expected = BoolArray::from_indices(10, [7], Validity::NonNullable).into_array();

        assert_arrays_eq!(expected, result);

        Ok(())
    }
}
