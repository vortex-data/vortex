// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
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
pub fn validate_replaces_null_views() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let views = Buffer::<BinaryView>::copy_from(vec![
        BinaryView::new_inlined(b"ololo"),
        BinaryView::new_ref(13, *b"AAAA", 0xDEAD_BEEF, 0xF000_0000),
    ]);
    let buffers: Arc<[ByteBuffer]> = Arc::new([]);
    let dtype = DType::Utf8(Nullability::Nullable);
    let buffer = BitBuffer::from_iter([true, false]);
    let validity = Validity::from_bit_buffer(buffer, Nullability::Nullable);

    let replaced =
        VarBinViewData::validate_and_fix(views.clone(), &buffers, &dtype, &validity, &mut ctx)?;
    assert_eq!(replaced[0], views[0]);
    assert_eq!(replaced[1], BinaryView::empty_view());

    let replaced =
        VarBinViewData::validate_and_fix(views, &buffers, &dtype, &Validity::AllInvalid, &mut ctx)?;
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
    let session = array_session();
    let array = VarBinViewArray::try_new(
        views.clone(),
        buffers,
        dtype.clone(),
        validity,
        &mut session.create_execution_ctx(),
    )?;

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

/// Validates `views` into a single data buffer as nullable UTF-8.
fn validate_utf8_views(
    buffer: &[u8],
    views: Vec<BinaryView>,
    validity: Validity,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let buffers: Arc<[ByteBuffer]> = Arc::new([ByteBuffer::from(buffer.to_vec())]);
    let views = Buffer::copy_from(views);
    let dtype = DType::Utf8(Nullability::Nullable);

    VarBinViewData::validate(&views, &buffers, &dtype, &validity, &mut ctx)
}

/// Makes a view of `buffer[start..end]` at buffer index 0.
fn view_of(buffer: &[u8], start: usize, end: usize) -> BinaryView {
    BinaryView::make_view(&buffer[start..end], 0, u32::try_from(start).unwrap())
}

#[test]
pub fn validate_multibyte_utf8_views() -> VortexResult<()> {
    let first = "zażółć gęślą jaźń";
    let buffer = format!("{first}łódź pod mostem");
    let buffer = buffer.as_bytes();

    let views = vec![
        view_of(buffer, 0, first.len()),
        view_of(buffer, first.len(), buffer.len()),
        BinaryView::new_inlined("żółw".as_bytes()),
    ];

    validate_utf8_views(buffer, views, Validity::AllValid)
}

#[rstest]
#[case::whole(0, 18, true)]
#[case::ascii_middle(2, 16, true)]
#[case::starts_inside_char(1, 18, false)]
#[case::ends_inside_char(0, 17, false)]
pub fn validate_view_char_boundaries(
    #[case] start: usize,
    #[case] end: usize,
    #[case] valid: bool,
) {
    // "ż" is two bytes, so the buffer is 18 bytes. The first view covers the whole buffer, so the
    // second view is checked against a buffer range that is valid UTF-8 as a whole.
    let buffer = "żaaaaaaaaaaaaaaż".as_bytes();
    let views = vec![
        view_of(buffer, 0, buffer.len()),
        view_of(buffer, start, end),
    ];

    assert_eq!(
        validate_utf8_views(buffer, views, Validity::AllValid).is_ok(),
        valid
    );
}

#[test]
pub fn validate_ignores_invalid_utf8_outside_valid_views() -> VortexResult<()> {
    let buffer = b"valid string one\xffvalid string two\xff\xfe\xfd garbage bytes";

    let views = vec![
        view_of(buffer, 0, 16),
        view_of(buffer, 17, 33),
        view_of(buffer, 33, buffer.len()),
    ];
    let validity = Validity::from_bit_buffer(
        BitBuffer::from_iter([true, true, false]),
        Nullability::Nullable,
    );

    validate_utf8_views(buffer, views, validity)
}

#[test]
pub fn validate_rejects_invalid_utf8_in_valid_views() {
    let buffer = b"valid string\xffmore bytes";
    let views = vec![view_of(buffer, 0, buffer.len())];
    assert!(validate_utf8_views(buffer, views, Validity::AllValid).is_err());

    let views = vec![BinaryView::new_inlined(b"ab\xff")];
    assert!(validate_utf8_views(buffer, views, Validity::AllValid).is_err());
}
