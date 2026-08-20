// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::slice::SliceReduce;
use crate::buffer::BufferHandle;

impl SliceReduce for Bool {
    fn slice(array: ArrayView<'_, Bool>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let (byte_start, meta) = array.meta.slice(range.clone());
        let byte_end = byte_start + meta.byte_len();

        let bits = if let Some(host) = array.bits.as_host_opt() {
            BufferHandle::new_host(host.slice_unaligned(byte_start..byte_end))
        } else {
            array.bits.slice(byte_start..byte_end)
        };
        let validity = array.validity()?.slice(range)?;

        let array = BoolArray::try_new_from_handle(bits, meta.offset(), meta.len(), validity)?;

        Ok(Some(array.into_array()))
    }
}
