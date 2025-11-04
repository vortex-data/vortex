// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{ArrayRef, GenericByteViewArray};
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_vector::binaryview::{BinaryType, BinaryViewVector, StringType};

use crate::arrow::IntoArrow;

macro_rules! impl_varbin {
    ($T:ty, $A:ty) => {
        impl IntoArrow<ArrayRef> for BinaryViewVector<$T> {
            fn into_arrow(self) -> VortexResult<ArrayRef> {
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
                    GenericByteViewArray::<$A>::new_unchecked(
                        views,
                        buffers,
                        validity.into_arrow()?,
                    )
                };
                Ok(Arc::new(array))
            }
        }
    };
}

impl_varbin!(BinaryType, arrow_array::types::BinaryViewType);
impl_varbin!(StringType, arrow_array::types::StringViewType);
