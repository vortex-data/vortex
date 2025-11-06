// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::ToPrimitive;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, IntegerPType, PTypeDowncastExt, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::binaryview::{
    BinaryType, BinaryView, BinaryViewType, BinaryViewVector, StringType,
};

use crate::ArrayRef;
use crate::arrays::{VarBinArray, VarBinVTable};
use crate::execution::{BatchKernel, BatchKernelRef, BindCtx, MaskExecution};
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<VarBinVTable> for VarBinVTable {
    fn bind(
        array: &VarBinArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;
        let offsets = ctx.bind(array.offsets(), None)?;

        match array.dtype() {
            DType::Utf8(_) => Ok(Box::new(VarBinKernel::<StringType>::new(
                offsets,
                array.bytes().clone(),
                validity,
                mask,
            ))),
            DType::Binary(_) => Ok(Box::new(VarBinKernel::<BinaryType>::new(
                offsets,
                array.bytes().clone(),
                validity,
                mask,
            ))),
            _ => unreachable!("invalid DType for VarBinArray {}", array.dtype()),
        }
    }
}

struct VarBinKernel<V> {
    offsets: BatchKernelRef,
    bytes: ByteBuffer,
    validity: MaskExecution,
    selection: MaskExecution,
    _type: PhantomData<V>,
}

impl<V> VarBinKernel<V> {
    fn new(
        offsets: BatchKernelRef,
        bytes: ByteBuffer,
        validity: MaskExecution,
        selection: MaskExecution,
    ) -> Self {
        Self {
            offsets,
            bytes,
            validity,
            selection,
            _type: PhantomData,
        }
    }
}

impl<V: BinaryViewType> BatchKernel for VarBinKernel<V> {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        let offsets = self.offsets.execute()?.into_primitive();

        match_each_integer_ptype!(offsets.ptype(), |T| {
            let pvec = offsets.downcast::<T>();
            // NOTE: discard the validity because offsets must be non-nullable
            let (offsets, _) = pvec.into_parts();
            let first = offsets[0];

            let lens: Buffer<u32> = offsets
                .iter()
                .copied()
                .skip(1)
                .scan(first, |prev, next| {
                    let len = (next - *prev)
                        .to_u32()
                        .vortex_expect("offset must map to u32");
                    *prev = next;
                    Some(len)
                })
                .collect();

            let selection = self.selection.execute()?;

            let views = match selection {
                Mask::AllFalse(_) => Buffer::empty(),
                Mask::AllTrue(_) => make_views::<T>(offsets.as_ref(), lens, &self.bytes),
                Mask::Values(values) => {
                    make_views_filtered::<T>(offsets.as_ref(), lens, values.indices(), &self.bytes)
                }
            };

            let validity = self.validity.execute()?;

            vortex_ensure!(
                validity.len() == views.len(),
                "mismatched validity and views length"
            );

            // SAFETY: views were constructed in the loop above to point at valid data from
            //  the buffer. Validity was checked immediately above to be of the appropriate length.
            Ok(Vector::from(unsafe {
                BinaryViewVector::<V>::new_unchecked(
                    views,
                    Arc::new(Box::new([self.bytes.clone()])),
                    validity,
                )
            }))
        })
    }
}

// Returns a set of views
fn make_views<OffsetType: IntegerPType>(
    offsets: &[OffsetType],
    lens: Buffer<u32>,
    bytes: &[u8],
) -> Buffer<BinaryView> {
    std::iter::zip(offsets, lens)
        .map(|(offset, len)| {
            let offset = offset.to_u32().vortex_expect("offset must fit in u32");
            let bytes = &bytes[offset as usize..(offset + len) as usize];
            if len as usize <= BinaryView::MAX_INLINED_SIZE {
                BinaryView::new_inlined(bytes)
            } else {
                BinaryView::make_view(bytes, 0, offset)
            }
        })
        .collect()
}

/// Only make views for values at the given `indices`
fn make_views_filtered<OffsetType: IntegerPType>(
    offsets: &[OffsetType],
    lens: Buffer<u32>,
    indices: &[usize],
    bytes: &[u8],
) -> Buffer<BinaryView> {
    indices
        .iter()
        .copied()
        .map(|index| {
            let offset = offsets[index]
                .to_u32()
                .vortex_expect("offset must fit in u32");
            let len = lens[index];
            let bytes = &bytes[offset as usize..(offset + len) as usize];
            if len as usize <= BinaryView::MAX_INLINED_SIZE {
                BinaryView::new_inlined(bytes)
            } else {
                BinaryView::make_view(bytes, 0, offset)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use vortex_dtype::{DType, Nullability};

    use crate::IntoArray;
    use crate::arrays::builder::VarBinBuilder;
    use crate::arrays::{BoolArray, VarBinArray};

    #[fixture]
    fn strings() -> VarBinArray {
        let mut strings = VarBinBuilder::<u32>::with_capacity(5);
        strings.append_value("inlined");
        strings.append_null();
        strings.append_value("large string 1");
        strings.append_value("large string 2");
        strings.append_value("large string 3");
        strings.finish(DType::Utf8(Nullability::Nullable))
    }

    #[rstest]
    fn test_bind(strings: VarBinArray) {
        // Attempt to bind with a full selection.
        let strings_vec = strings
            .bind(None, &mut ())
            .unwrap()
            .execute()
            .unwrap()
            .into_string();
        assert_eq!(strings_vec.get_ref(0), Some("inlined"));
        assert_eq!(strings_vec.get_ref(1), None);
        assert_eq!(strings_vec.get_ref(2), Some("large string 1"));
        assert_eq!(strings_vec.get_ref(3), Some("large string 2"));
        assert_eq!(strings_vec.get_ref(4), Some("large string 3"));
    }

    #[rstest]
    fn test_bind_with_selection(strings: VarBinArray) {
        let selection = BoolArray::from_iter([false, true, false, true, true]).into_array();
        let strings_vec = strings
            .bind(Some(&selection), &mut ())
            .unwrap()
            .execute()
            .unwrap()
            .into_string();

        assert_eq!(strings_vec.get_ref(0), None);
        assert_eq!(strings_vec.get_ref(1), Some("large string 2"));
        assert_eq!(strings_vec.get_ref(2), Some("large string 3"));
    }
}
