// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_array::execution::ExecutionCtx;
use vortex_array::vtable::OperatorVTable;
use vortex_array::{Array, ArrayOperator};
use vortex_buffer::Buffer;
use vortex_dtype::{IntegerPType, PType, PTypeDowncastExt};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::primitive::{PVector, PrimitiveVector};
use vortex_vector::{Vector, VectorOps, match_each_integer_pvector};

use crate::{ALPRDArray, ALPRDFloat, ALPRDVTable};

impl OperatorVTable<ALPRDVTable> for ALPRDVTable {
    fn execute_batch(
        array: &ALPRDArray,
        selection: &Mask,
        ctx: &mut dyn ExecutionCtx,
    ) -> VortexResult<Vector> {
        let ptype = array.dtype().as_ptype();
        let patches = array.left_parts_patches();

        let (left_parts, validity) = array
            .left_parts()
            .execute_batch(selection, ctx)?
            .into_primitive()
            .downcast::<u16>()
            .into_parts();
        let mut left_parts = left_parts.into_mut();

        let left_parts_dict = array.left_parts_dictionary();

        let right_parts = array
            .right_parts()
            .execute_batch(selection, ctx)?
            .into_primitive();

        match (ptype, patches) {
            (PType::F32, Some(patches)) => {
                let (mut right_parts, _) = right_parts.into_mut().downcast::<u32>().into_parts();
                let n_patches = patches.indices().len();
                let mask = Mask::new_true(n_patches);
                let patch_values_pvec = patches
                    .values()
                    .execute_batch(&mask, ctx)?
                    .into_primitive()
                    .downcast::<u16>();
                let patch_indices_pvec = patches
                    .indices()
                    .execute_batch(&mask, ctx)?
                    .into_primitive();
                match_each_integer_pvector!(patch_indices_pvec, |pvec| {
                    alp_rd_decode_in_place_patched::<f32, _>(
                        left_parts.as_mut(),
                        left_parts_dict.as_ref(),
                        array.right_bit_width(),
                        right_parts.as_mut(),
                        pvec.as_ref(),
                        patch_values_pvec.as_ref(),
                    );

                    // Cast the right parts to f32
                    let right_parts: Buffer<f32> = unsafe { std::mem::transmute(right_parts) };

                    Ok(PrimitiveVector::from(PVector::new(right_parts, validity)).into_vec())
                })
            }

            (PType::F32, None) => {
                let (mut right_parts, _) = right_parts.into_mut().downcast::<u32>().into_parts();
                alp_rd_decode_in_place::<f32>(
                    left_parts.as_mut(),
                    left_parts_dict.as_ref(),
                    array.right_bit_width(),
                    right_parts.as_mut(),
                );

                // Cast the right parts to f32
                let right_parts: Buffer<f32> = unsafe { std::mem::transmute(right_parts) };

                Ok(PrimitiveVector::from(PVector::new(right_parts, validity)).into_vec())
            }

            (PType::F64, Some(patches)) => {
                let (mut right_parts, _) = right_parts.into_mut().downcast::<u64>().into_parts();
                let n_patches = patches.indices().len();
                let mask = Mask::new_true(n_patches);
                let patch_values_pvec = patches
                    .values()
                    .execute_batch(&mask, ctx)?
                    .into_primitive()
                    .downcast::<u16>();
                let patch_indices_pvec = patches
                    .indices()
                    .execute_batch(&mask, ctx)?
                    .into_primitive();
                match_each_integer_pvector!(patch_indices_pvec, |pvec| {
                    alp_rd_decode_in_place_patched::<f64, _>(
                        left_parts.as_mut(),
                        left_parts_dict.as_ref(),
                        array.right_bit_width(),
                        right_parts.as_mut(),
                        pvec.as_ref(),
                        patch_values_pvec.as_ref(),
                    );

                    // Cast the right parts to f32
                    let right_parts: Buffer<f64> = unsafe { std::mem::transmute(right_parts) };

                    Ok(PrimitiveVector::from(PVector::new(right_parts, validity)).into_vec())
                })
            }
            (PType::F64, None) => {
                let (mut right_parts, _) = right_parts.into_mut().downcast::<u64>().into_parts();
                alp_rd_decode_in_place::<f64>(
                    left_parts.as_mut(),
                    left_parts_dict.as_ref(),
                    array.right_bit_width(),
                    right_parts.as_mut(),
                );

                // Cast the right parts to f32
                let right_parts: Buffer<f32> = unsafe { std::mem::transmute(right_parts) };

                Ok(PrimitiveVector::from(PVector::new(right_parts, validity)).into_vec())
            }
            _ => unreachable!("ALP-RD arrays only support f32/f64"),
        }
    }
}

/// Insert the left-parts (after decoding using the dictionary) into
/// the right_parts in-place.
///
/// The stored value will be as an u32/u64, which can be transmuted to
/// another vector of this type.
fn alp_rd_decode_in_place<T: ALPRDFloat>(
    left_parts: &mut [u16],
    left_parts_dict: &[u16],
    right_bw: u8,
    right_parts: &mut [T::UINT],
) {
    // Decode all the left-parts first
    for part in left_parts.as_mut().iter_mut() {
        *part = left_parts_dict[*part as usize];
    }

    // Insert left-parts into right parts with shift
    let shift = right_bw as usize;
    for (&left, right) in iter::zip(left_parts.as_ref(), right_parts.as_mut()) {
        *right = (<T as ALPRDFloat>::from_u16(left)) << shift | *right;
    }
}

/// Insert the left-parts (after decoding using the dictionary) into
/// the right_parts in-place.
///
/// The stored value will be as an u32/u64, which can be transmuted to
/// another vector of this type.
fn alp_rd_decode_in_place_patched<T: ALPRDFloat, Index: IntegerPType>(
    left_parts: &mut [u16],
    left_parts_dict: &[u16],
    right_bw: u8,
    right_parts: &mut [T::UINT],
    patch_indices: &[Index],
    patch_values: &[u16],
) {
    // Decode all the left-parts first
    for part in left_parts.as_mut().iter_mut() {
        *part = left_parts_dict[*part as usize];
    }

    // Apply patches
    for (&index, &value) in iter::zip(patch_indices.as_ref(), patch_values.as_ref()) {
        left_parts[index.as_()] = value;
    }

    // Insert left-parts into right parts with shift
    let shift = right_bw as usize;
    for (&left, right) in iter::zip(left_parts.as_ref(), right_parts.as_mut()) {
        *right = <T as ALPRDFloat>::from_u16(left) << shift | *right;
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::execution::DummyExecutionCtx;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::PTypeDowncastExt;
    use vortex_mask::Mask;

    use crate::RDEncoder;

    #[test]
    fn test_execute_batch() {
        let real_doubles = buffer![
            f64::from_bits(0x7F7F_AAAAAAAAAAAA),
            f64::from_bits(0x1313_AAAAAAAAAAAA),
            f64::from_bits(0x1010_AAAAAAAAAAAA),
            f64::from_bits(0x1010_BBBBBBBBBBBB),
            f64::from_bits(0x1313_BBBBBBBBBBBB),
            f64::from_bits(0x1313_CCCCCCCCCCCC),
        ]
        .into_array()
        .to_primitive();

        let encoded = RDEncoder::new(real_doubles.as_slice::<f64>()).encode(&real_doubles);

        let decoded = encoded
            .execute_batch(&Mask::new_true(6), &mut DummyExecutionCtx)
            .unwrap()
            .into_primitive()
            .downcast::<f64>();

        assert_eq!(
            decoded.as_ref(),
            &[
                f64::from_bits(0x7F7F_AAAAAAAAAAAA),
                f64::from_bits(0x1313_AAAAAAAAAAAA),
                f64::from_bits(0x1010_AAAAAAAAAAAA),
                f64::from_bits(0x1010_BBBBBBBBBBBB),
                f64::from_bits(0x1313_BBBBBBBBBBBB),
                f64::from_bits(0x1313_CCCCCCCCCCCC),
            ]
        );
    }

    #[test]
    fn test_execute_batch_patches() {}
}
