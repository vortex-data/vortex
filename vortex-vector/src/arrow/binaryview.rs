// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::arrow::nulls_to_mask;
use crate::binaryview::BinaryType;
use crate::binaryview::BinaryViewType;
use crate::binaryview::BinaryViewVector;
use crate::binaryview::StringType;
use crate::binaryview::view::BinaryView;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::GenericByteViewArray;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::vortex_err;

macro_rules! impl_binaryview_to_arrow {
    ($T:ty, $A:ty) => {
        impl TryFrom<BinaryViewVector<$T>> for ArrayRef {
            type Error = VortexError;

            fn try_from(value: BinaryViewVector<$T>) -> Result<Self, Self::Error> {
                let (views, buffers, validity) = value.into_parts();

                let views = Buffer::<u128>::from_byte_buffer(views.into_byte_buffer())
                    .into_arrow_scalar_buffer();
                let buffers: Vec<_> = buffers
                    .iter()
                    .cloned()
                    .map(|b| b.into_arrow_buffer())
                    .collect();

                // SAFETY: our own guarantees are the same as Arrow's guarantees for BinaryViewArray
                let array = unsafe {
                    GenericByteViewArray::<$A>::new_unchecked(views, buffers, validity.into())
                };
                Ok(Arc::new(array))
            }
        }
    };
}

impl_binaryview_to_arrow!(BinaryType, arrow_array::types::BinaryViewType);
impl_binaryview_to_arrow!(StringType, arrow_array::types::StringViewType);

macro_rules! impl_binaryview_from_arrow {
    ($T:ty, $A:ty) => {
        impl TryFrom<ArrayRef> for BinaryViewVector<$T> {
            type Error = VortexError;

            fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
                let array = value
                    .as_any()
                    .downcast_ref::<GenericByteViewArray<$A>>()
                    .ok_or_else(|| {
                        vortex_err!(
                            "expected GenericByteViewArray<{}>, got {}",
                            stringify!($A),
                            value.data_type()
                        )
                    })?;

                // Convert views from Arrow's u128 representation to BinaryView
                let arrow_views = array.views();
                let views = Buffer::<BinaryView>::from_byte_buffer(
                    Buffer::<u128>::from_arrow_scalar_buffer(arrow_views.clone())
                        .into_byte_buffer(),
                );

                // Convert buffers
                let buffers: Box<[ByteBuffer]> = array
                    .data_buffers()
                    .iter()
                    .map(|b| {
                        ByteBuffer::from_arrow_buffer(
                            b.clone(),
                            vortex_buffer::Alignment::of::<u8>(),
                        )
                    })
                    .collect();

                let validity = nulls_to_mask(array.nulls(), array.len());

                // SAFETY: Arrow's GenericByteViewArray maintains the same invariants as our BinaryViewVector
                Ok(unsafe { BinaryViewVector::new_unchecked(views, Arc::new(buffers), validity) })
            }
        }
    };
}

impl_binaryview_from_arrow!(BinaryType, arrow_array::types::BinaryViewType);
impl_binaryview_from_arrow!(StringType, arrow_array::types::StringViewType);
