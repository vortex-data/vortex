// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use fsst::Decompressor;
use num_traits::AsPrimitive;
use vortex_array::Array;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::build_views::BinaryView;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::matchers::Exact;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::FSSTArray;
use crate::FSSTVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<FSSTVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FSSTFilterKernel)]);

#[derive(Debug)]
struct FSSTFilterKernel;

impl ExecuteParentKernel<FSSTVTable> for FSSTFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &FSSTArray,
        parent: &FilterArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        let mask_values = match parent.filter_mask() {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => v,
        };

        // We filter the uncompressed lengths
        let uncompressed_lens = array
            .uncompressed_lengths()
            .filter(parent.filter_mask().clone())?
            .execute::<Canonical>(ctx)?
            .into_primitive();

        // Extract the filtered validity
        let validity = match array.codes().validity().filter(parent.filter_mask())? {
            Validity::NonNullable | Validity::AllValid => {
                Mask::new_true(parent.filter_mask().true_count())
            }
            Validity::AllInvalid => Mask::new_false(parent.filter_mask().true_count()),
            Validity::Array(a) => a.execute::<Mask>(ctx)?,
        };

        // First we unpack the codes VarBinArray to get access to the raw data.
        let codes_data = array.codes().bytes();
        let codes_offsets = array
            .codes()
            .offsets()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(ctx)?
            .to_buffer::<u32>();

        let decompressor = array.decompressor();

        let (views, buffer) = match_each_integer_ptype!(uncompressed_lens.ptype(), |S| {
            fsst_decode::<S>(
                decompressor,
                codes_data,
                &codes_offsets,
                mask_values,
                &validity,
                &uncompressed_lens.to_buffer::<S>(),
            )
        });

        // SAFETY: FSST already validates the bytes for binary/UTF-8.
        let canonical = unsafe {
            Canonical::VarBinView(VarBinViewArray::new_unchecked(
                views,
                Arc::from(vec![buffer]),
                array.dtype().clone(),
                Validity::from_mask(validity, array.dtype().nullability()),
            ))
        };

        Ok(Some(canonical))
    }
}

fn fsst_decode<S: IntegerPType + AsPrimitive<usize> + AsPrimitive<u32>>(
    decompressor: Decompressor,
    codes_data: &[u8],
    codes_offsets: &[u32],
    filter_mask: &MaskValues,
    filtered_validity: &Mask,
    filtered_uncompressed_lengths: &[S],
) -> (Buffer<BinaryView>, ByteBuffer) {
    let total_uncompressed_size: usize = filtered_uncompressed_lengths
        .iter()
        .map(|x| <S as AsPrimitive<usize>>::as_(*x))
        .sum();

    // We allocate an extra 7 bytes per the FSST decompressor's requirement for padding.
    let mut uncompressed = ByteBufferMut::with_capacity(total_uncompressed_size + 7);
    let mut spare_capacity = uncompressed.spare_capacity_mut();

    match filtered_validity {
        Mask::AllTrue(_) => {
            for &idx in filter_mask.indices() {
                let start = codes_offsets[idx] as usize;
                let end = codes_offsets[idx + 1] as usize;
                let compressed_slice = &codes_data[start..end];

                let uncompressed_len =
                    decompressor.decompress_into(compressed_slice, spare_capacity);
                spare_capacity = &mut spare_capacity[uncompressed_len..];
            }
        }
        Mask::AllFalse(_) => {
            // Nothing to decompress - all values are null with length 0
        }
        Mask::Values(values) => {
            for (filtered_idx, (idx, is_valid)) in filter_mask
                .indices()
                .iter()
                .copied()
                .zip(values.bit_buffer().iter())
                .enumerate()
            {
                if is_valid {
                    let start = codes_offsets[idx] as usize;
                    let end = codes_offsets[idx + 1] as usize;
                    let compressed_slice = &codes_data[start..end];

                    let uncompressed_len =
                        decompressor.decompress_into(compressed_slice, spare_capacity);
                    spare_capacity = &mut spare_capacity[uncompressed_len..];
                } else {
                    // We advance the output buffer to make it faster to assemble views below.
                    spare_capacity =
                        &mut spare_capacity[filtered_uncompressed_lengths[filtered_idx].as_()..];
                }
            }
        }
    }

    unsafe { uncompressed.set_len(total_uncompressed_size) };
    let uncompressed = uncompressed.freeze();
    let uncompressed_slice = uncompressed.as_ref();

    // Loop over the uncompressed lengths to construct the BinaryViews.
    let mut views = BufferMut::<BinaryView>::with_capacity(filtered_uncompressed_lengths.len());
    let mut offset = 0u32;
    for len in filtered_uncompressed_lengths {
        let view = BinaryView::make_view(
            &uncompressed_slice[offset as usize..][..len.as_()],
            0u32,
            offset,
        );
        offset += <S as AsPrimitive<u32>>::as_(*len);
        unsafe { views.push_unchecked(view) };
    }
    unsafe { views.set_len(filtered_uncompressed_lengths.len()) };

    (views.freeze(), uncompressed)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Array;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FilterArray;
    use vortex_array::arrays::builder::VarBinBuilder;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::FSSTVTable;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn build_test_fsst_array() -> ArrayRef {
        let mut builder = VarBinBuilder::<i32>::with_capacity(10);
        builder.append_value(b"hello world");
        builder.append_value(b"foo bar baz");
        builder.append_value(b"testing fsst compression");
        builder.append_value(b"another string here");
        builder.append_value(b"the quick brown fox");
        builder.append_value(b"jumps over the lazy dog");
        builder.append_value(b"abcdefghijklmnop");
        builder.append_value(b"qrstuvwxyz");
        builder.append_value(b"0123456789");
        builder.append_value(b"final string");
        let input = builder.finish(DType::Utf8(Nullability::NonNullable));

        let compressor = fsst_train_compressor(&input);
        fsst_compress(input, &compressor).into_array()
    }

    #[test]
    fn test_fsst_filter_simple() -> VortexResult<()> {
        let fsst_array = build_test_fsst_array();
        assert!(fsst_array.is::<FSSTVTable>());
        assert_eq!(fsst_array.len(), 10);

        // Filter 1/5 elements (every 5th element: indices 0 and 5)
        let mask = Mask::from_iter([
            true, false, false, false, false, true, false, false, false, false,
        ]);

        // Create FilterArray and execute
        let filter_array = FilterArray::new(fsst_array.clone(), mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        // Compare with filtering the canonical VarBinView.
        let expected = fsst_array.filter(mask)?;

        assert_eq!(result.len(), 2);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }

    #[test]
    fn test_fsst_filter_every_other() -> VortexResult<()> {
        let fsst_array = build_test_fsst_array();

        // Filter every other element
        let mask = Mask::from_iter([
            true, false, true, false, true, false, true, false, true, false,
        ]);

        let filter_array = FilterArray::new(fsst_array.clone(), mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        let expected = fsst_array.filter(mask)?;

        assert_eq!(result.len(), 5);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }

    #[test]
    fn issues_6034_test_fsst_filter_with_nulls_and_special_chars() -> VortexResult<()> {
        //
        // Test case with special characters and nulls
        // Values: ["", "", "", "", "", "", "", "", "", "", "", ",", "A<<<<<<<", "", "", "", "", null, null, null, null, null, null]
        // Mask: only the last element is selected (true at index 22)
        let mut builder = VarBinBuilder::<i32>::with_capacity(23);
        // 11 empty strings
        for _ in 0..11 {
            builder.append_value(b"");
        }
        // ","
        builder.append_value(b",");
        // "A<<<<<<<"
        builder.append_value(b"A<<<<<<<");
        // 4 more empty strings
        for _ in 0..4 {
            builder.append_value(b"");
        }
        // 6 nulls
        for _ in 0..6 {
            builder.append_null();
        }
        let input = builder.finish(DType::Utf8(Nullability::Nullable));

        let compressor = fsst_train_compressor(&input);
        let fsst_array: ArrayRef = fsst_compress(input.clone(), &compressor).into_array();

        // Filter: only select the last element (index 22)
        let mut mask = vec![false; 22];
        mask.push(true);
        let mask = Mask::from_iter(mask);

        let filter_array = FilterArray::new(fsst_array.clone(), mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        let expected = input.filter(mask)?;

        assert_eq!(result.len(), 1);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }

    #[test]
    fn filter_only_null() -> VortexResult<()> {
        let mut builder = VarBinBuilder::<i32>::with_capacity(3);
        builder.append_null();
        builder.append_value(b"A");
        builder.append_null();

        let input = builder.finish(DType::Utf8(Nullability::Nullable));

        let compressor = fsst_train_compressor(&input);
        let fsst_array: ArrayRef = fsst_compress(input.clone(), &compressor).into_array();

        let mask = Mask::from_iter([true, false, true]);

        let filter_array = FilterArray::new(fsst_array.clone(), mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        let expected = input.filter(mask)?;

        assert_eq!(result.len(), 2);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }
}
