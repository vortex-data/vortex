// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-width byte strings, which order lexicographically by byte.
//!
//! Both fixed-size binary arrays and fixed-size lists of non-nullable `u8` hold them. The latter
//! are the storage of byte-string extension types such as UUID, so order-based aggregates can
//! support them without implementing general list comparison.

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

/// Whether `dtype` is a fixed-size list of non-nullable `u8`.
pub(crate) fn is_fixed_size_bytes(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::FixedSizeList(element_dtype, ..)
            if matches!(element_dtype.as_ref(), DType::Primitive(PType::U8, Nullability::NonNullable))
    )
}

/// The values of an array of fixed-width byte strings, each one a slice of one host buffer.
pub(crate) struct FixedSizeBytes {
    bytes: ByteBuffer,
    size: usize,
    len: usize,
}

impl FixedSizeBytes {
    /// Decode the elements of `array`, which must be a fixed-size list of bytes.
    pub(crate) fn try_new(
        array: &FixedSizeListArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let elements = array.elements().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(Self {
            bytes: elements.buffer_handle().to_host_sync(),
            size: array.list_size() as usize,
            len: array.len(),
        })
    }

    /// The values of a fixed-size binary array.
    pub(crate) fn from_fixed_size_binary(array: &FixedSizeBinaryArray) -> Self {
        Self {
            bytes: array.values(),
            size: array.byte_width() as usize,
            len: array.len(),
        }
    }

    /// Every value in order, including the values of null rows.
    pub(crate) fn values(&self) -> impl Iterator<Item = &[u8]> {
        let bytes = self.bytes.as_slice();
        (0..self.len).map(move |index| &bytes[index * self.size..(index + 1) * self.size])
    }
}
