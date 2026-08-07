// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewData;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

#[test]
pub fn varbin_view() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]);
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]),
        &mut ctx
    );
}

#[test]
pub fn slice_array() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
            .slice(1..2)
            .unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world this is a long string"]),
        &mut ctx
    );
}

#[test]
pub fn flatten_array() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr = VarBinViewArray::from_iter_str(["string1", "string2"]);
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["string1", "string2"]),
        &mut ctx
    );
}

#[test]
pub fn binary_view_size_and_alignment() {
    assert_eq!(size_of::<BinaryView>(), 16);
    assert_eq!(align_of::<BinaryView>(), 16);
}

#[test]
pub fn replace_invalid_views() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let views = Buffer::<BinaryView>::copy_from(vec![
        BinaryView::new_inlined(b"ololo"),
        BinaryView::new_ref(13, *b"AAAA", 0xDEAD_BEEF, 0xF000_0000),
    ]);
    let buffer = BitBuffer::from_iter([true, false]);
    let validity = Validity::from_bit_buffer(buffer, Nullability::Nullable);

    let replaced = VarBinViewData::replace_invalid_views(views.clone(), &validity, &mut ctx)?;
    assert_eq!(replaced[0], views[0]);
    assert_eq!(replaced[1], BinaryView::empty_view());

    let replaced = VarBinViewData::replace_invalid_views(views, &Validity::AllInvalid, &mut ctx)?;
    assert!(
        replaced
            .iter()
            .all(|view| *view == BinaryView::empty_view())
    );
    Ok(())
}

#[test]
pub fn deserialize_null_views() -> VortexResult<()> {
    let views = Buffer::<BinaryView>::copy_from(vec![
        BinaryView::new_ref(14, *b"hell", 0, 0),
        BinaryView::new_ref(13, *b"AAAA", 0xDEAD_BEEF, 0xF000_0000),
    ]);
    let buffers = Arc::new([ByteBuffer::from(b"hello world ololo".to_vec())]);
    let dtype = DType::Utf8(Nullability::Nullable);
    let buffer = BitBuffer::from_iter([true, false]);
    let validity = Validity::from_bit_buffer(buffer, Nullability::Nullable);
    let array = VarBinViewArray::try_new(views.clone(), buffers, dtype.clone(), validity)?;

    let session = array_session();
    let array_ctx = ArrayContext::empty();
    let serialized =
        array
            .clone()
            .into_array()
            .serialize(&array_ctx, &session, &SerializeOptions::default())?;

    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(
        &dtype,
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &session,
    )?;

    let decoded = decoded
        .as_opt::<VarBinView>()
        .ok_or_else(|| vortex_err!("expected VarBinView"))?;
    assert_eq!(decoded.views()[0], views[0]);
    assert_eq!(decoded.views()[1], BinaryView::empty_view());
    Ok(())
}
