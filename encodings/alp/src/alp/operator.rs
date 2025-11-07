// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::mem::transmute;

use vortex_array::execution::ExecutionCtx;
use vortex_array::patches::Patches;
use vortex_array::vtable::OperatorVTable;
use vortex_array::{Array, ArrayOperator};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{IntegerPType, PTypeDowncastExt, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::primitive::{PVector, PrimitiveVector, PrimitiveVectorMut};
use vortex_vector::{Vector, VectorMutOps, VectorOps};

use crate::{ALPArray, ALPFloat, ALPVTable, Exponents, match_each_alp_float_ptype};

impl OperatorVTable<ALPVTable> for ALPVTable {
    fn execute_batch(
        array: &ALPArray,
        selection: &Mask,
        ctx: &mut dyn ExecutionCtx,
    ) -> VortexResult<Vector> {
        let exponents = array.exponents();
        let encoded_vector = array
            .encoded()
            .execute_batch(selection, ctx)?
            .into_primitive();

        let decoded_vector = match_each_alp_float_ptype!(array.ptype(), |Float| {
            type Int = <Float as ALPFloat>::ALPInt;
            let (buffer, validity) = encoded_vector.downcast::<Int>().into_parts();
            let decoded = decode_slice::<Float>(buffer.into_mut(), exponents);
            unsafe { PrimitiveVector::from(PVector::new_unchecked(decoded, validity)) }
        });

        // Apply the exponents to the entire Vector.
        // NOTE: some items will fail. We allow this to happen silently.
        // The final result is going to be either f32 or f64.

        match array.patches() {
            None => Ok(decoded_vector.into_vec()),
            Some(patches) => apply_patches(decoded_vector.into_mut(), patches, ctx),
        }
    }
}

// Apply the slice this way instead
fn decode_slice<P: ALPFloat>(mut values: BufferMut<P::ALPInt>, exponents: Exponents) -> Buffer<P> {
    // How much does this work
    P::decode_slice_inplace(values.as_mut_slice(), exponents);
    // SAFETY: ALPFloat and the corresponding ALPInt have the same size and alignment
    let ints_buffer: BufferMut<P> = unsafe { transmute(values) };
    ints_buffer.freeze()
}

// Apply patches to the ALP result
// TODO(aduffy): take Patches by value once execute_batch can be taken by value
fn apply_patches(
    encoded: PrimitiveVectorMut,
    patches: &Patches,
    ctx: &mut dyn ExecutionCtx,
) -> VortexResult<Vector> {
    let n_patches = patches.indices().len();

    // We decode all patches. They should be a very small fraction of the initial vector.
    let values = patches
        .values()
        .execute_batch(&Mask::new_true(n_patches), ctx)?
        .into_primitive();
    let indices = patches
        .indices()
        .execute_batch(&Mask::new_true(n_patches), ctx)?
        .into_primitive();

    // Apply inner
    fn apply_inner<Value: ALPFloat, Index: IntegerPType>(
        values: &[Value],
        indices: &[Index],
        output: &mut [Value],
    ) {
        for (&value, index) in iter::zip(values, indices) {
            let index = index.as_();
            output[index] = value;
        }
    }

    match_each_alp_float_ptype!(encoded.ptype(), |Value| {
        let values_ref = values.downcast::<Value>();
        let mut pvector = encoded.downcast::<Value>();
        match_each_integer_ptype!(indices.ptype(), |Index| {
            let indices_ref = indices.downcast::<Index>();
            apply_inner::<Value, Index>(
                values_ref.as_ref(),
                indices_ref.as_ref(),
                pvector.as_mut(),
            );

            Ok(PrimitiveVector::from(pvector.freeze()).into_vec())
        })
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::execution::DummyExecutionCtx;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::PTypeDowncastExt;
    use vortex_mask::Mask;

    use crate::alp_encode;

    #[test]
    fn test_execute_batch_no_patches() {
        let floats = buffer![1f32, 2f32, 3f32].into_array().to_primitive();
        let encoded = alp_encode(&floats, None).unwrap();

        // No patches
        assert!(encoded.patches().is_none());

        let floats_vector = encoded
            .execute_batch(&Mask::new_true(3), &mut DummyExecutionCtx)
            .unwrap()
            .into_primitive()
            .downcast::<f32>();

        assert_eq!(floats_vector.get(0).copied(), Some(1f32));
        assert_eq!(floats_vector.get(1).copied(), Some(2f32));
        assert_eq!(floats_vector.get(2).copied(), Some(3f32));
    }

    #[test]
    fn test_execute_batch_patches() {
        // second item will force a patch
        let floats = buffer![1f32, 200000000f32, 3f32, 4f32, 5000000000019f32]
            .into_array()
            .to_primitive();
        let encoded = alp_encode(&floats, None).unwrap();

        // Has patches for the two outlier values
        assert!(encoded.patches().is_some());

        let floats_vector = encoded
            .execute_batch(&Mask::new_true(5), &mut DummyExecutionCtx)
            .unwrap()
            .into_primitive()
            .downcast::<f32>();
        assert_eq!(floats_vector.get(0).copied(), Some(1f32));
        assert_eq!(floats_vector.get(1).copied(), Some(200000000f32));
        assert_eq!(floats_vector.get(2).copied(), Some(3f32));
        assert_eq!(floats_vector.get(3).copied(), Some(4f32));
        assert_eq!(floats_vector.get(4).copied(), Some(5000000000019f32));
    }
}
