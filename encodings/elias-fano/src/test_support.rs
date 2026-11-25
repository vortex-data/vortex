// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_session::registry::ReadContext;

use crate::SESSION;

/// `values` as a primitive array of `ptype`, which must hold every one of them.
pub(super) fn primitive(ptype: PType, values: &[u64]) -> PrimitiveArray {
    match_each_unsigned_integer_ptype!(ptype, |T| {
        PrimitiveArray::from_iter(values.iter().map(|&value| AsPrimitive::<T>::as_(value)))
    })
}

pub(super) fn serde_roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let mut concat = ByteBufferMut::empty();
    for buffer in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        concat.extend_from_slice(buffer.as_ref());
    }

    SerializedArray::try_from(concat.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )
}
