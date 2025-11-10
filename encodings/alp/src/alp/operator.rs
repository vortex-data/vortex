// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::transmute;

use vortex_array::ArrayOperator;
use vortex_array::execution::ExecutionCtx;
use vortex_array::vtable::OperatorVTable;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::primitive::{PVector, PrimitiveVector};
use vortex_vector::{Vector, VectorMutOps, VectorOps};

use crate::{
    ALPArray, ALPFloat, ALPVTable, Exponents, apply_patches_in_place, match_each_alp_float_ptype,
};

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

        match array.patches() {
            None => Ok(decoded_vector.into_vec()),
            Some(patches) => {
                let mut patched = decoded_vector.into_mut();
                apply_patches_in_place(&mut patched, patches, ctx)?;
                Ok(patched.freeze().into_vec())
            }
        }
    }
}

// Apply the slice this way instead
fn decode_slice<P: ALPFloat>(mut values: BufferMut<P::ALPInt>, exponents: Exponents) -> Buffer<P> {
    P::decode_slice_inplace(values.as_mut_slice(), exponents);
    // SAFETY: ALPFloat and the corresponding ALPInt have the same size and alignment
    let ints_buffer: BufferMut<P> = unsafe { transmute(values) };
    ints_buffer.freeze()
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
