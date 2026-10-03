// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use rstest::rstest;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use super::OffsetMetadata;
use super::Utf8OffsetColumn;
use super::Utf8OffsetValues;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::IntoArray as _;
use crate::VortexSessionExecute as _;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArraySlotsExt as _;
use crate::arrays::varbin::VarBinData;
use crate::arrays::varbin::VarBinSlots;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::buffer::DeviceBuffer;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_rows;
use crate::validity::Validity;

#[derive(Debug, PartialEq, Eq, Hash)]
pub(in crate::scalar_fn::unstable::row::types::element) struct Unreadable {
    pub(in crate::scalar_fn::unstable::row::types::element) len: usize,
}

impl DeviceBuffer for Unreadable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.len
    }

    fn alignment(&self) -> Alignment {
        Alignment::of::<u8>()
    }

    fn copy_to_host_sync(&self, _alignment: Alignment) -> VortexResult<ByteBuffer> {
        vortex_bail!("payload host access rejected")
    }

    fn copy_to_host(
        &self,
        _alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        vortex_bail!("payload host access rejected")
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        Arc::new(Self { len: range.len() })
    }

    fn aligned(self: Arc<Self>, _alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        Ok(self)
    }
}

pub(in crate::scalar_fn::unstable::row::types::element) fn unreadable_offset(
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let offsets =
        PrimitiveArray::new(Buffer::from(vec![0u32, 13]), Validity::NonNullable).into_array();
    let bytes = BufferHandle::new_device(Arc::new(Unreadable { len: 13 }));
    let dtype = DType::Utf8(validity.nullability());
    let data =
        VarBinData::try_build_from_handle(offsets.clone(), bytes, dtype.clone(), validity.clone())?;
    let slots = VarBinSlots {
        offsets,
        validity: matches!(validity, Validity::AllInvalid)
            .then(|| ConstantArray::new(false, 1).into_array()),
    }
    .into_slots();

    Ok(
        VarBinArray::try_from_parts(ArrayParts::new(VarBin, dtype, 1, data).with_slots(slots))?
            .into_array(),
    )
}

#[test]
fn decode_reuses_buffers_and_retains_owners() -> VortexResult<()> {
    let array = VarBinArray::from_iter_nonnull(
        ["", "a longer string", "héllo"],
        DType::Utf8(Nullability::NonNullable),
    );
    let mut ctx = array_session().create_execution_ctx();
    let offsets = array
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .to_buffer::<u32>();
    let bytes = array.bytes_handle().try_to_host_sync()?;
    let decoded = Utf8OffsetColumn::decode(array.into_array(), &mut ctx)?;

    assert_eq!(decoded.metadata.offsets.as_ptr(), offsets.as_ptr());
    assert_eq!(decoded.bytes.as_ptr(), bytes.as_ptr());
    drop(offsets);
    drop(bytes);
    let view = Utf8OffsetColumn::view(&decoded);
    for (index, expected) in ["", "a longer string", "héllo"].into_iter().enumerate() {
        assert_eq!(Utf8OffsetColumn::get(&decoded, index), expected);
        assert_eq!(Utf8OffsetColumn::get_from_view(&view, index), expected);
        // SAFETY: the enumeration has one index for each retained row.
        assert_eq!(
            unsafe { Utf8OffsetColumn::get_from_view_unchecked(&view, index) },
            expected
        );
    }

    Ok(())
}

#[test]
fn decode_preserves_sliced_offsets() -> VortexResult<()> {
    let array = VarBinArray::from_iter_nonnull(
        ["before", "héllo", "a longer string", "after"],
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
    .slice(1..3)?;
    let mut ctx = array_session().create_execution_ctx();
    let decoded = Utf8OffsetColumn::decode(array, &mut ctx)?;

    assert_eq!(Utf8OffsetColumn::get(&decoded, 0), "héllo");
    assert_eq!(Utf8OffsetColumn::get(&decoded, 1), "a longer string");
    Ok(())
}

#[test]
fn decode_reads_no_all_null_payload() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decoded = Utf8OffsetColumn::decode(unreadable_offset(Validity::AllInvalid)?, &mut ctx)?;
    assert_eq!(Utf8OffsetColumn::get(&decoded, 0), "");
    assert!(Utf8OffsetColumn::decode(unreadable_offset(Validity::NonNullable)?, &mut ctx).is_err());
    Ok(())
}

#[rstest]
#[case::past_payload(vec![0u32, 14], 1, 1)]
#[case::decreasing(vec![13u32, 0], 1, 1)]
#[case::missing_end(vec![0u32], 1, 1)]
#[case::mask_length(vec![0u32, 13], 2, 1)]
fn decode_rejects_malformed_offsets(
    #[case] offsets: Vec<u32>,
    #[case] mask_rows: usize,
    #[case] rows: usize,
) {
    assert!(
        OffsetMetadata::from_parts(Buffer::from(offsets), Mask::new_true(mask_rows), 13, rows)
            .is_err()
    );
}

#[rstest]
#[case::valid(true)]
#[case::null(false)]
fn decode_validates_utf8_only_on_readable_rows(#[case] valid: bool) -> VortexResult<()> {
    let metadata =
        OffsetMetadata::from_parts(Buffer::from(vec![0u32, 1]), Mask::new(1, valid), 1, 1)?;
    let decoded = Utf8OffsetValues::from_parts(metadata, ByteBuffer::from(vec![0xff]));
    if valid {
        assert!(decoded.is_err());
    } else {
        assert_eq!(Utf8OffsetColumn::get(&decoded?, 0), "");
    }
    Ok(())
}

#[test]
fn decode_rejects_dtype_layout_and_offset_width() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    assert!(
        Utf8OffsetColumn::decode(PrimitiveArray::from_iter([1u32]).into_array(), &mut ctx).is_err()
    );
    assert!(
        Utf8OffsetColumn::decode(
            VarBinViewArray::from_iter_str(["text"]).into_array(),
            &mut ctx
        )
        .is_err()
    );
    let wide = VarBinArray::try_new(
        PrimitiveArray::from_iter([0u64, 4]).into_array(),
        ByteBuffer::from(b"text".to_vec()),
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
    )?;
    assert!(Utf8OffsetColumn::decode(wide.into_array(), &mut ctx).is_err());
    Ok(())
}

#[derive(Clone)]
struct Prefix;

impl RowFn for Prefix {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.offset_prefix");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &EmptyOptions,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(Utf8OffsetColumn,), bool>(|(value,)| value.starts_with("pre"))
    }
}

#[rstest]
#[case::empty(vec![], vec![])]
#[case::all_valid(vec![Some("prefix"), Some("héllo")], vec![Some(true), Some(false)])]
#[case::partially_valid(vec![Some("prefix"), None, Some("other")], vec![Some(true), None, Some(false)])]
#[case::all_null(vec![None, None], vec![None, None])]
fn row_fn_propagates_strict_nulls(
    #[case] values: Vec<Option<&str>>,
    #[case] expected: Vec<Option<bool>>,
) -> VortexResult<()> {
    let input = VarBinArray::from_iter(values, DType::Utf8(Nullability::Nullable)).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], input.len());
    let mut ctx = array_session().create_execution_ctx();
    let output = execute_rows(&Prefix, &EmptyOptions, &args, &mut ctx)?;
    assert_arrays_eq!(
        &output,
        &BoolArray::from_iter(expected).into_array(),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::inline("prefix")]
#[case::referenced("prefix with a longer value")]
fn row_fn_preserves_batch_constants(#[case] value: &str) -> VortexResult<()> {
    let input = ConstantArray::new(value, 17).into_array();
    let args = VecExecutionArgs::new(vec![input], 17);
    let mut ctx = array_session().create_execution_ctx();
    let output = execute_rows(&Prefix, &EmptyOptions, &args, &mut ctx)?;
    assert_arrays_eq!(
        &output,
        &ConstantArray::new(true, 17).into_array(),
        &mut ctx
    );
    Ok(())
}
