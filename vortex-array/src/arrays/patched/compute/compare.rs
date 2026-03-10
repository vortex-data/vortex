// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_error::{VortexExpect, VortexResult};

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PatchedVTable;
use crate::arrays::bool::BoolArrayParts;
use crate::arrays::patched::patch_lanes;
use crate::arrays::primitive::NativeValue;
use crate::builtins::ArrayBuiltins;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;

impl CompareKernel for PatchedVTable {
    fn compare(
        lhs: &Self::Array,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        let result = lhs
            .inner
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

        fn apply<V: NativePType, F>(
            bits: &mut BitBufferMut,
            lane_offsets: &[u32],
            indices: &[u16],
            values: &[V],
            constant: V,
            cmp: F,
        ) -> VortexResult<()>
        where
            F: Fn(V, V) -> bool,
        {
            let n_lanes = patch_lanes::<V>();

            for index in 0..(lane_offsets.len() - 1) {
                let chunk = index / n_lanes;

                let lane_start = lane_offsets[index] as usize;
                let lane_end = lane_offsets[index + 1] as usize;

                for (&patch_index, &patch_value) in std::iter::zip(
                    &indices[lane_start..lane_end],
                    &values[lane_start..lane_end],
                ) {
                    let bit_index = chunk * 1024 + patch_index as usize;
                    if cmp(patch_value, constant) {
                        bits.set(bit_index)
                    } else {
                        bits.unset(bit_index)
                    }
                }
            }

            Ok(())
        }

        let lane_offsets = lhs.lane_offsets.as_host().reinterpret::<u32>();
        let indices = lhs.indices.as_host().reinterpret::<u16>();

        match_each_native_ptype!(lhs.values_ptype, |V| {
            let values = lhs.values.as_host().reinterpret::<V>();
            let constant = constant
                .as_primitive()
                .as_::<V>()
                .vortex_expect("compare constant not null");

            match operator {
                CompareOperator::Eq => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| NativeValue(l) == NativeValue(r),
                    )?;
                }
                CompareOperator::NotEq => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| NativeValue(l) != NativeValue(r),
                    )?;
                }
                CompareOperator::Gt => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| NativeValue(l) > NativeValue(r),
                    )?;
                }
                CompareOperator::Gte => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| NativeValue(l) >= NativeValue(r),
                    )?;
                }
                CompareOperator::Lt => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| NativeValue(l) < NativeValue(r),
                    )?;
                }
                CompareOperator::Lte => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| NativeValue(l) <= NativeValue(r),
                    )?;
                }
            }
        });

        // SAFETY: thing
        let result = unsafe { BoolArray::new_unchecked(bits.freeze(), validity) };
        Ok(Some(result.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PatchedArray;
    use crate::arrays::PatchedVTable;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
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

        let lhs = PatchedArray::from_array_and_patches(lhs, &patches, &mut ctx).unwrap();

        let rhs = ConstantArray::new(u32::MAX, 512).into_array();

        let result =
            <PatchedVTable as CompareKernel>::compare(&lhs, &rhs, CompareOperator::Eq, &mut ctx)
                .unwrap()
                .unwrap();

        let expected =
            BoolArray::from_indices(512, [509, 510, 511], Validity::NonNullable).into_array();

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
        let lhs = PatchedArray::from_array_and_patches(lhs, &patches, &mut ctx)?;

        let rhs = ConstantArray::new(subnormal, 512).into_array();

        let result =
            <PatchedVTable as CompareKernel>::compare(&lhs, &rhs, CompareOperator::Eq, &mut ctx)?
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
        let lhs = PatchedArray::from_array_and_patches(lhs, &patches, &mut ctx)?;

        let rhs = ConstantArray::new(0.0f32, 10).into_array();

        let result =
            <PatchedVTable as CompareKernel>::compare(&lhs, &rhs, CompareOperator::Eq, &mut ctx)?
                .unwrap();

        let expected = BoolArray::from_indices(10, [7], Validity::NonNullable).into_array();

        assert_arrays_eq!(expected, result);

        Ok(())
    }
}
