// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use fsst::Decompressor;
use num_traits::AsPrimitive;
use vortex_array::Array;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VectorExecutor;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::mask::MaskExecutor;
use vortex_array::matchers::Exact;
use vortex_array::validity::Validity;
use vortex_array::vectors::VectorIntoArray;
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
use vortex_vector::binaryview::BinaryVector;
use vortex_vector::binaryview::BinaryView;
use vortex_vector::binaryview::StringVector;

use crate::FSSTArray;
use crate::FSSTVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<FSSTVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FSSTFilterKernel)]);

#[derive(Debug)]
struct FSSTFilterKernel;

impl ExecuteParentKernel<FSSTVTable> for FSSTFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&FilterVTable)
    }

    // TODO(joe); remove Vector usage internally?
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
            .execute(ctx)?
            .into_primitive();

        // Extract the filtered validity
        let validity = match array.codes().validity().filter(parent.filter_mask())? {
            Validity::NonNullable | Validity::AllValid => {
                Mask::new_true(parent.filter_mask().true_count())
            }
            Validity::AllInvalid => Mask::new_false(parent.filter_mask().true_count()),
            Validity::Array(a) => a.execute_mask(ctx)?,
        };

        // First we unpack the codes VarBinArray to get access to the raw data.
        let codes_data = array.codes().bytes();
        let codes_offsets = array
            .codes()
            .offsets()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))?
            .execute(ctx)?
            .into_primitive()
            .buffer::<u32>();

        let decompressor = array.decompressor();

        let (views, buffer) = match_each_integer_ptype!(uncompressed_lens.ptype(), |S| {
            fsst_decode::<S>(
                decompressor,
                codes_data,
                &codes_offsets,
                mask_values,
                &validity,
                &uncompressed_lens.buffer::<S>(),
            )
        });

        let dtype = array.dtype();
        let canonical = match dtype {
            DType::Binary(_) => unsafe {
                BinaryVector::new_unchecked(views, Arc::new(vec![buffer].into()), validity)
            }
            .into_array(array.dtype())
            .to_canonical(),
            DType::Utf8(_) => unsafe {
                StringVector::new_unchecked(views, Arc::new(vec![buffer].into()), validity)
            }
            .into_array(array.dtype())
            .to_canonical(),
            _ => unreachable!("Not a supported FSST DType"),
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
            // Nothing to decompress
            unsafe { uncompressed.set_len(0) };
            return (Buffer::empty(), uncompressed.freeze());
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
