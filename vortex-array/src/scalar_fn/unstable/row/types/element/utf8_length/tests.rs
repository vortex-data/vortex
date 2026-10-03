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
use crate::scalar_fn::unstable::row::types::element::utf8_offset::tests::Unreadable;
use crate::scalar_fn::unstable::row::types::element::utf8_offset::tests::unreadable_offset;
use crate::validity::Validity;

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

fn input(layout: Layout, values: Vec<Option<&str>>) -> ArrayRef {
    match layout {
        Layout::Offsets => {
            VarBinArray::from_iter(values, DType::Utf8(Nullability::Nullable)).into_array()
        }
        Layout::Views => VarBinViewArray::from_iter_nullable_str(values).into_array(),
    }
}

#[test]
fn metadata_decode_never_requests_payload() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let column =
        Utf8OffsetLengthColumn::decode(unreadable_offset(Validity::NonNullable)?, &mut ctx)?;
    assert_eq!(Utf8OffsetLengthColumn::get(&column, 0), 13);

    let views = Buffer::from(vec![BinaryView::make_view(b"1234567890123", 0, 0)]);
    let array = VarBinViewArray::new_handle(
        BufferHandle::new_host(views.into_byte_buffer()),
        Arc::from(vec![BufferHandle::new_device(Arc::new(Unreadable {
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
fn unspecified_null_header_stays_null() -> VortexResult<()> {
    let views = Buffer::from(vec![
        BinaryView::make_view(b"a longer string", 0, 0),
        BinaryView::from(u128::MAX),
    ]);
    let array = VarBinViewArray::new_handle(
        BufferHandle::new_host(views.into_byte_buffer()),
        Arc::from(vec![BufferHandle::new_device(Arc::new(Unreadable {
            len: 15,
        }))]),
        DType::Utf8(Nullability::Nullable),
        Validity::from_iter([true, false]),
    )
    .into_array();
    let mut ctx = array_session().create_execution_ctx();
    let column = Utf8ViewLengthColumn::decode(array.clone(), &mut ctx)?;
    assert_eq!(Utf8ViewLengthColumn::get(&column, 1), u64::from(u32::MAX));
    let args = VecExecutionArgs::new(vec![array], 2);
    let output = execute_rows(&Threshold(Layout::Views), &EmptyOptions, &args, &mut ctx)?;
    assert_arrays_eq!(
        &output,
        &BoolArray::from_iter([Some(true), None]).into_array(),
        &mut ctx
    );
    Ok(())
}

#[test]
fn metadata_decode_rejects_wrong_dtype_and_layout() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([13u32]).into_array();
    assert!(Utf8OffsetLengthColumn::decode(primitive.clone(), &mut ctx).is_err());
    assert!(Utf8ViewLengthColumn::decode(primitive, &mut ctx).is_err());
    assert!(
        Utf8ViewLengthColumn::decode(unreadable_offset(Validity::NonNullable)?, &mut ctx).is_err()
    );
    let views = VarBinViewArray::from_iter_str(["text"]).into_array();
    assert!(Utf8OffsetLengthColumn::decode(views, &mut ctx).is_err());
    Ok(())
}

#[rstest]
#[case::offsets(Layout::Offsets)]
#[case::views(Layout::Views)]
fn access_paths_use_utf8_byte_lengths(#[case] layout: Layout) -> VortexResult<()> {
    let values = input(
        layout,
        vec![
            Some("outside"),
            Some(""),
            Some("héllo"),
            Some("a longer string"),
            Some("outside"),
        ],
    );
    let values = values.slice(1..4)?;
    let mut ctx = array_session().create_execution_ctx();
    let expected = [0, 6, 15];
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
                assert_eq!(
                    unsafe { Utf8OffsetLengthColumn::get_from_view_unchecked(&view, index) },
                    expected
                );
            }
        }
        Layout::Views => {
            let column = Utf8ViewLengthColumn::decode(values, &mut ctx)?;
            let view = Utf8ViewLengthColumn::view(&column);
            for (index, expected) in expected.into_iter().enumerate() {
                assert_eq!(Utf8ViewLengthColumn::get(&column, index), expected);
                assert_eq!(Utf8ViewLengthColumn::get_from_view(&view, index), expected);
                // SAFETY: there is one expected length for each retained row.
                assert_eq!(
                    unsafe { Utf8ViewLengthColumn::get_from_view_unchecked(&view, index) },
                    expected
                );
            }
        }
    }
    Ok(())
}

#[rstest]
#[case::empty(vec![], vec![])]
#[case::all_valid(vec![Some(""), Some("a longer string")], vec![Some(false), Some(true)])]
#[case::partial(vec![Some("héllo"), None, Some("a longer string")], vec![Some(false), None, Some(true)])]
#[case::all_null(vec![None, None], vec![None, None])]
fn row_fn_matches_scalar_lengths(
    #[case] values: Vec<Option<&str>>,
    #[case] expected: Vec<Option<bool>>,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    for layout in [Layout::Offsets, Layout::Views] {
        let input = input(layout, values.clone());
        let args = VecExecutionArgs::new(vec![input.clone()], input.len());
        let output = execute_rows(&Threshold(layout), &EmptyOptions, &args, &mut ctx)?;
        assert_arrays_eq!(
            &output,
            &BoolArray::from_iter(expected.clone()).into_array(),
            &mut ctx
        );
    }
    Ok(())
}

#[test]
fn metadata_constants_use_one_value() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let value = ConstantArray::new("a longer string", 17).into_array();
    assert_eq!(
        Utf8OffsetLengthColumn::decode_constant(value.clone(), &mut ctx)?,
        15
    );
    assert_eq!(
        Utf8ViewLengthColumn::decode_constant(value.clone(), &mut ctx)?,
        15
    );
    for layout in [Layout::Offsets, Layout::Views] {
        let args = VecExecutionArgs::new(vec![value.clone()], 17);
        let output = execute_rows(&Threshold(layout), &EmptyOptions, &args, &mut ctx)?;
        assert_arrays_eq!(
            &output,
            &ConstantArray::new(true, 17).into_array(),
            &mut ctx
        );
    }
    Ok(())
}
