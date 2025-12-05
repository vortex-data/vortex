// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::GenericByteViewArray;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_vector::binaryview::BinaryType;
use vortex_vector::binaryview::BinaryView;
use vortex_vector::binaryview::BinaryViewVector;
use vortex_vector::binaryview::StringType;

use crate::arrow::IntoArrow;
use crate::arrow::IntoVector;
use crate::arrow::nulls_to_mask;

macro_rules! impl_binaryview_to_arrow {
    ($T:ty, $A:ty) => {
        impl IntoArrow for BinaryViewVector<$T> {
            type Output = ArrayRef;

            fn into_arrow(self) -> VortexResult<Self::Output> {
                let (views, buffers, validity) = self.into_parts();

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
        impl IntoVector for &GenericByteViewArray<$A> {
            type Output = BinaryViewVector<$T>;

            fn into_vector(self) -> VortexResult<Self::Output> {
                // Convert views from Arrow's u128 representation to BinaryView
                let arrow_views = self.views();
                let views = Buffer::<BinaryView>::from_byte_buffer(
                    Buffer::<u128>::from_arrow_scalar_buffer(arrow_views.clone())
                        .into_byte_buffer(),
                );

                // Convert buffers
                let buffers: Box<[ByteBuffer]> = self
                    .data_buffers()
                    .iter()
                    .map(|b| {
                        ByteBuffer::from_arrow_buffer(
                            b.clone(),
                            vortex_buffer::Alignment::of::<u8>(),
                        )
                    })
                    .collect();

                let validity = nulls_to_mask(self.nulls(), self.len());

                // SAFETY: Arrow's GenericByteViewArray maintains the same invariants as our BinaryViewVector
                Ok(unsafe { BinaryViewVector::new_unchecked(views, Arc::new(buffers), validity) })
            }
        }
    };
}

impl_binaryview_from_arrow!(BinaryType, arrow_array::types::BinaryViewType);
impl_binaryview_from_arrow!(StringType, arrow_array::types::StringViewType);
