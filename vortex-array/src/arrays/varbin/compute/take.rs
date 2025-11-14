// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{BitBufferMut, BufferMut, ByteBufferMut};
use vortex_dtype::{DType, IntegerPType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_mask::Mask;

use crate::arrays::varbin::VarBinArray;
use crate::arrays::{PrimitiveArray, VarBinVTable};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for VarBinVTable {
    fn take(&self, array: &VarBinArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let offsets = array.offsets().to_primitive();
        let data = array.bytes();
        let indices = indices.to_primitive();
        match_each_integer_ptype!(offsets.ptype(), |O| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                Ok(take(
                    array
                        .dtype()
                        .clone()
                        .union_nullability(indices.dtype().nullability()),
                    offsets.as_slice::<O>(),
                    data.as_slice(),
                    indices.as_slice::<I>(),
                    array.validity_mask(),
                    indices.validity_mask(),
                )?
                .into_array())
            })
        })
    }
}

register_kernel!(TakeKernelAdapter(VarBinVTable).lift());

fn take<I: IntegerPType, O: IntegerPType>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    validity_mask: Mask,
    indices_validity_mask: Mask,
) -> VortexResult<VarBinArray> {
    if !validity_mask.all_true() || !indices_validity_mask.all_true() {
        return Ok(take_nullable(
            dtype,
            offsets,
            data,
            indices,
            validity_mask,
            indices_validity_mask,
        ));
    }

    let mut new_offsets = BufferMut::with_capacity(indices.len() + 1);
    new_offsets.push(O::zero());
    let mut current_offset = O::zero();

    for &idx in indices {
        let idx = idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", idx));

        // Bounds check to prevent out-of-bounds access
        if idx >= offsets.len() - 1 {
            vortex_panic!(
                "Index {} out of bounds for offsets with length {}",
                idx,
                offsets.len() - 1
            );
        }

        let start = offsets[idx];
        let stop = offsets[idx + 1];

        // Check for valid offset range
        if stop < start {
            vortex_panic!("Invalid offset range: stop ({:?}) < start ({:?})", stop, start);
        }

        // Use checked arithmetic to prevent overflow
        let length = stop - start;
        current_offset = current_offset
            .checked_add(&length)
            .unwrap_or_else(|| {
                vortex_panic!(
                    "Offset overflow when computing new offset: {} + {} exceeds maximum value for type",
                    current_offset, length
                )
            });
        new_offsets.push(current_offset);
    }

    let mut new_data = ByteBufferMut::with_capacity(
        current_offset
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize"),
    );

    for idx in indices {
        let idx = idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", idx));
        let start = offsets[idx]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        let stop = offsets[idx + 1]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        new_data.extend_from_slice(&data[start..stop]);
    }

    let array_validity = Validity::from(dtype.nullability());

    // Safety:
    // All variants of VarBinArray are satisfied here.
    unsafe {
        Ok(VarBinArray::new_unchecked(
            PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable).into_array(),
            new_data.freeze(),
            dtype,
            array_validity,
        ))
    }
}

fn take_nullable<I: IntegerPType, O: IntegerPType>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    data_validity: Mask,
    indices_validity: Mask,
) -> VarBinArray {
    let mut new_offsets = BufferMut::with_capacity(indices.len() + 1);
    new_offsets.push(O::zero());
    let mut current_offset = O::zero();

    let mut validity_buffer = BitBufferMut::with_capacity(indices.len());

    // Convert indices once and store valid ones with their positions
    let mut valid_indices = Vec::with_capacity(indices.len());

    // First pass: calculate offsets and validity
    for (idx, data_idx) in indices.iter().enumerate() {
        if !indices_validity.value(idx) {
            validity_buffer.append(false);
            new_offsets.push(current_offset);
            continue;
        }
        let data_idx_usize = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        // Bounds check to prevent out-of-bounds access
        if data_idx_usize >= offsets.len() - 1 {
            vortex_panic!(
                "Index {} out of bounds for offsets with length {}",
                data_idx_usize,
                offsets.len() - 1
            );
        }

        if data_validity.value(data_idx_usize) {
            validity_buffer.append(true);
            let start = offsets[data_idx_usize];
            let stop = offsets[data_idx_usize + 1];

            // Check for valid offset range
            if stop < start {
                vortex_panic!(
                    "Invalid offset range: stop ({:?}) < start ({:?})",
                    stop,
                    start
                );
            }

            // Use checked arithmetic to prevent overflow
            let length = stop - start;
            current_offset = current_offset
                .checked_add(&length)
                .unwrap_or_else(|| {
                    vortex_panic!(
                        "Offset overflow when computing new offset: {} + {} exceeds maximum value for type",
                        current_offset, length
                    )
                });
            new_offsets.push(current_offset);
            valid_indices.push(data_idx_usize);
        } else {
            validity_buffer.append(false);
            new_offsets.push(current_offset);
        }
    }

    let mut new_data = ByteBufferMut::with_capacity(
        current_offset
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize"),
    );

    // Second pass: copy data for valid indices only
    for data_idx in valid_indices {
        let start = offsets[data_idx]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        let stop = offsets[data_idx + 1]
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        new_data.extend_from_slice(&data[start..stop]);
    }

    let array_validity = Validity::from(validity_buffer.freeze());

    // Safety:
    // All variants of VarBinArray are satisfied here.
    unsafe {
        VarBinArray::new_unchecked(
            PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable).into_array(),
            new_data.freeze(),
            dtype,
            array_validity,
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability};

    use crate::Array;
    use crate::arrays::{PrimitiveArray, VarBinArray};
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::IntoArray;

    #[test]
    fn test_null_take() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            take(arr.as_ref(), idx1.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            take(arr.as_ref(), idx2.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("test"), Some("data"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(["single"].map(Some), DType::Utf8(Nullability::NonNullable)))]
    fn test_take_varbin_conformance(#[case] array: VarBinArray) {
        test_take_conformance(array.as_ref());
    }

    #[test]
    #[should_panic(expected = "Offset overflow when computing new offset")]
    fn test_take_overflow_protection() {
        // Regression test for fuzzer crash: test handling of very large offsets
        // that could cause overflow when accumulated
        let data = Buffer::copy_from(b"test data with some content that is fairly long");

        // Create offsets that when accumulated could overflow u8
        // This tests the checked_add logic
        // With u8, max value is 255. We create offsets where accumulating lengths exceeds this.
        let offsets: PrimitiveArray = PrimitiveArray::from_iter(vec![0u8, 200u8, 250u8]);

        let varbin = unsafe {
            VarBinArray::new_unchecked(
                offsets.into_array(),
                data,
                DType::Binary(Nullability::NonNullable),
                Validity::NonNullable,
            )
        };

        // Try to take indices that would accumulate offsets beyond u8::MAX
        // First element has length 200, second has length 50, total 250 which is fine
        // But if we take first element twice: 200 + 200 = 400 > 255, causing overflow
        let indices: PrimitiveArray = PrimitiveArray::from_iter(vec![0u32, 0u32]);

        // Should panic due to offset overflow when computing new offset
        let _ = take(varbin.as_ref(), indices.as_ref());
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds for offsets with length 2")]
    fn test_take_bounds_check() {
        // Test bounds checking prevents out-of-bounds access
        let data = Buffer::copy_from(b"hello world");
        let offsets: PrimitiveArray = PrimitiveArray::from_iter(vec![0u32, 5u32, 11u32]);

        let varbin = unsafe {
            VarBinArray::new_unchecked(
                offsets.into_array(),
                data,
                DType::Binary(Nullability::NonNullable),
                Validity::NonNullable,
            )
        };

        // Try to take with an index that's out of bounds
        // We have 2 elements (indices 0 and 1), so index 5 is out of bounds
        let indices: PrimitiveArray = PrimitiveArray::from_iter(vec![5u32]);

        // Should panic due to out of bounds index
        let _ = take(varbin.as_ref(), indices.as_ref());
    }

    #[test]
    #[should_panic(expected = "Invalid offset range")]
    fn test_take_invalid_offset_range() {
        // Test that invalid offset ranges (where stop < start) are caught
        let data = Buffer::copy_from(b"hello world");

        // Create invalid offsets where stop < start
        let offsets: PrimitiveArray = PrimitiveArray::from_iter(vec![0u32, 10u32, 5u32]);

        let varbin = unsafe {
            VarBinArray::new_unchecked(
                offsets.into_array(),
                data,
                DType::Binary(Nullability::NonNullable),
                Validity::NonNullable,
            )
        };

        // Try to take the second element (index 1) which has stop=5 < start=10
        let indices: PrimitiveArray = PrimitiveArray::from_iter(vec![1u32]);

        // Should panic due to invalid offset range
        let _ = take(varbin.as_ref(), indices.as_ref());
    }
}
