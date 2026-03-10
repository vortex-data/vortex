// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::bool::BoolArrayParts;
use crate::arrays::patched::patch_lanes;
use crate::arrays::{BoolArray, ConstantArray, PatchedVTable};
use crate::builtins::ArrayBuiltins;
use crate::dtype::NativePType;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::{ArrayRef, Canonical, ExecutionCtx, IntoArray, match_each_unsigned_integer_ptype};
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

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

        match_each_unsigned_integer_ptype!(lhs.values_ptype, |V| {
            let values = lhs.values.as_host().reinterpret::<V>();
            let constant = constant
                .as_primitive()
                .as_::<V>()
                .expect("compare constant not null");

            match operator {
                CompareOperator::Eq => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| l == r,
                    )?;
                }
                CompareOperator::NotEq => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| l != r,
                    )?;
                }
                CompareOperator::Gt => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| l > r,
                    )?;
                }
                CompareOperator::Gte => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| l >= r,
                    )?;
                }
                CompareOperator::Lt => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| l < r,
                    )?;
                }
                CompareOperator::Lte => {
                    apply::<V, _>(
                        &mut bits,
                        lane_offsets,
                        indices,
                        values,
                        constant,
                        |l, r| l <= r,
                    )?;
                }
            }
        });

        // Stitch up final bool array with validity
        let result = unsafe { BoolArray::new_unchecked(bits.freeze(), validity) };
        Ok(Some(result.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use crate::arrays::{BoolArray, ConstantArray, PatchedArray, PatchedVTable, PrimitiveArray};
    use crate::patches::Patches;
    use crate::scalar_fn::fns::binary::CompareKernel;
    use crate::scalar_fn::fns::operators::CompareOperator;
    use crate::validity::Validity;
    use crate::{ExecutionCtx, IntoArray, LEGACY_SESSION, assert_arrays_eq};
    use vortex_buffer::buffer;

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
    fn test_subnormal() {

    }
}
