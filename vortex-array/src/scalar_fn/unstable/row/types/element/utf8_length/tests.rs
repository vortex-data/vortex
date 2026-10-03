// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use super::Utf8OffsetLengthColumn;
use super::Utf8ViewLengthColumn;
use crate::ArrayRef;
use crate::IntoArray as _;
use crate::VortexSessionExecute as _;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_rows;
use crate::scalar_fn::unstable::row::types::element::test_support::UnreadablePayload;
use crate::scalar_fn::unstable::row::types::element::test_support::unreadable_offset_array;
use crate::validity::Validity;

#[track_caller]
fn assert_threshold(layout: Layout, input: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
    let args = VecExecutionArgs::new(vec![input.clone()], input.len());
    let mut ctx = array_session().create_execution_ctx();
    let output = execute_rows(&Threshold(layout), &EmptyOptions, &args, &mut ctx)?;

    assert_arrays_eq!(&output, &expected, &mut ctx);

    Ok(())
}

/// The native representation selected explicitly by each test.
#[derive(Clone, Copy)]
enum Layout {
    Offsets,
    Views,
}

#[derive(Clone)]
struct Threshold(Layout);

impl RowFn for Threshold {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.utf8_length_threshold");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &EmptyOptions,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        match self.0 {
            Layout::Offsets => visitor.visit::<(Utf8OffsetLengthColumn,), bool>(|(len,)| len >= 13),
            Layout::Views => visitor.visit::<(Utf8ViewLengthColumn,), bool>(|(len,)| len >= 13),
        }
    }
}

/// Build nullable UTF-8 input in the representation under test.
fn input(layout: Layout, values: Vec<Option<&str>>) -> ArrayRef {
    match layout {
        Layout::Offsets => {
            VarBinArray::from_iter(values, DType::Utf8(Nullability::Nullable)).into_array()
        }
        Layout::Views => VarBinViewArray::from_iter_nullable_str(values).into_array(),
    }
}

#[test]
fn test_metadata_decode_never_requests_payload() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let column =
        Utf8OffsetLengthColumn::decode(unreadable_offset_array(Validity::NonNullable)?, &mut ctx)?;
    assert_eq!(Utf8OffsetLengthColumn::get(&column, 0), 13);

    let views = Buffer::from(vec![BinaryView::make_view(b"1234567890123", 0, 0)]);
    let array = VarBinViewArray::new_handle(
        BufferHandle::new_host(views.into_byte_buffer()),
        Arc::from(vec![BufferHandle::new_device(Arc::new(UnreadablePayload {
            len: 13,
        }))]),
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
    )
    .into_array();
    let column = Utf8ViewLengthColumn::decode(array, &mut ctx)?;
    assert_eq!(Utf8ViewLengthColumn::get(&column, 0), 13);

    Ok(())
}

#[test]
fn test_unspecified_null_header_stays_null() -> VortexResult<()> {
    let views = Buffer::from(vec![
        BinaryView::make_view(b"a longer string", 0, 0), // Valid row with unreadable payload.
        BinaryView::from(u128::MAX),                   // Null row with an unspecified header.
    ]);
    let array = VarBinViewArray::new_handle(
        BufferHandle::new_host(views.into_byte_buffer()),
        Arc::from(vec![BufferHandle::new_device(Arc::new(UnreadablePayload {
            len: 15,
        }))]),
        DType::Utf8(Nullability::Nullable),
        Validity::from_iter([true, false]),
    )
    .into_array();
    let mut ctx = array_session().create_execution_ctx();
    let column = Utf8ViewLengthColumn::decode(array.clone(), &mut ctx)?;
    assert_eq!(Utf8ViewLengthColumn::get(&column, 1), u64::from(u32::MAX));

    assert_threshold(
        Layout::Views,
        array,
        BoolArray::from_iter([Some(true), None]).into_array(),
    )
}

#[test]
fn test_metadata_decode_rejects_wrong_dtype_and_layout() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([13u32]).into_array();
    assert!(Utf8OffsetLengthColumn::decode(primitive.clone(), &mut ctx).is_err());
    assert!(Utf8ViewLengthColumn::decode(primitive, &mut ctx).is_err());
    let offsets = unreadable_offset_array(Validity::NonNullable)?;
    assert!(Utf8ViewLengthColumn::decode(offsets, &mut ctx).is_err());
    let views = VarBinViewArray::from_iter_str(["text"]).into_array();
    assert!(Utf8OffsetLengthColumn::decode(views, &mut ctx).is_err());

    Ok(())
}

#[rstest]
#[case::offsets(Layout::Offsets)]
#[case::views(Layout::Views)]
fn test_access_paths_use_utf8_byte_lengths(#[case] layout: Layout) -> VortexResult<()> {
    let values = input(
        layout,
        vec![
            Some("outside"),         // Excluded prefix.
            Some(""),                // Empty.
            Some("héllo"),           // Six UTF-8 bytes.
            Some("a longer string"), // Fifteen ASCII bytes.
            Some("outside"),         // Excluded suffix.
        ],
    );
    let values = values.slice(1..4)?;
    let mut ctx = array_session().create_execution_ctx();
    let expected = [
        0,  // Empty.
        6,  // UTF-8 byte length, rather than character count.
        15, // Long ASCII.
    ];

    match layout {
        Layout::Offsets => {
            let column = Utf8OffsetLengthColumn::decode(values, &mut ctx)?;
            let view = Utf8OffsetLengthColumn::view(&column);

            for (index, expected) in expected.into_iter().enumerate() {
                assert_eq!(Utf8OffsetLengthColumn::get(&column, index), expected);
                assert_eq!(
                    Utf8OffsetLengthColumn::get_from_view(&view, index),
                    expected
                );

                // SAFETY: there is one expected length for each retained row.
                let unchecked =
                    unsafe { Utf8OffsetLengthColumn::get_from_view_unchecked(&view, index) };
                assert_eq!(unchecked, expected);
            }
        }
        Layout::Views => {
            let column = Utf8ViewLengthColumn::decode(values, &mut ctx)?;
            let view = Utf8ViewLengthColumn::view(&column);

            for (index, expected) in expected.into_iter().enumerate() {
                assert_eq!(Utf8ViewLengthColumn::get(&column, index), expected);
                assert_eq!(Utf8ViewLengthColumn::get_from_view(&view, index), expected);

                // SAFETY: there is one expected length for each retained row.
                let unchecked =
                    unsafe { Utf8ViewLengthColumn::get_from_view_unchecked(&view, index) };
                assert_eq!(unchecked, expected);
            }
        }
    }

    Ok(())
}

#[rstest]
#[case::empty(vec![], vec![])]
#[case::all_valid(vec![Some(""), Some("a longer string")], vec![Some(false), Some(true)])]
#[case::partially_valid(vec![Some("héllo"), None, Some("a longer string")], vec![Some(false), None, Some(true)])]
#[case::all_null(vec![None, None], vec![None, None])]
fn test_row_fn_matches_scalar_lengths(
    #[values(Layout::Offsets, Layout::Views)] layout: Layout,
    #[case] values: Vec<Option<&str>>,
    #[case] expected: Vec<Option<bool>>,
) -> VortexResult<()> {
    assert_threshold(
        layout,
        input(layout, values),
        BoolArray::from_iter(expected).into_array(),
    )
}

#[rstest]
#[case::offsets(Layout::Offsets)]
#[case::views(Layout::Views)]
fn test_metadata_constants_use_one_value(#[case] layout: Layout) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let value = ConstantArray::new("a longer string", 17).into_array();
    let length = match layout {
        Layout::Offsets => Utf8OffsetLengthColumn::decode_constant(value.clone(), &mut ctx)?,
        Layout::Views => Utf8ViewLengthColumn::decode_constant(value.clone(), &mut ctx)?,
    };
    assert_eq!(length, 15);

    assert_threshold(layout, value, ConstantArray::new(true, 17).into_array())
}
