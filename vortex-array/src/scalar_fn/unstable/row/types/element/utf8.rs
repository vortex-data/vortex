// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 input columns for row functions.
//!
//! [`Utf8Column`] decodes a `Utf8` column once per batch and hands each row callback a `&str`.
//! A [`VarBin`] column is read through its own offsets when they fit in `u32` and every row, null
//! rows included, is valid UTF-8. Decoding then builds no string views. Every other input executes
//! to string views.

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::primitive::PrimitiveArrayExt as _;
use crate::arrays::varbin::VarBinArraySlotsExt as _;
use crate::arrays::varbin::offsets_tile_utf8;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewArrayExt as _;
use crate::arrays::varbinview::VarBinViewData;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::scalar::ScalarValue;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::ViewLen;

/// A UTF-8 input element that yields `&str` values.
///
/// A null row yields a valid but unspecified string: the empty string for view storage, or the
/// bytes stored at that row for offset storage. The executor masks every output computed from a
/// null row.
pub struct Utf8Column;

/// One decoded UTF-8 column.
///
/// The storage stays private so that safe code cannot build storage that skipped the validation
/// that unchecked row access relies on.
pub struct Utf8Values(Utf8Storage);

/// The two layouts a decoded column can keep. Only [`decode_utf8`] and [`decode_views`] build it.
enum Utf8Storage {
    /// String views whose every view addresses valid UTF-8 in `buffers`.
    Views {
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
    },
    /// One more offset than rows. The offsets never decrease, end within `bytes`, and delimit
    /// valid UTF-8 at every row, null rows included.
    Offsets {
        offsets: Buffer<u32>,
        bytes: ByteBuffer,
    },
}

/// A borrowed UTF-8 column prepared for a row loop.
#[derive(Clone, Copy)]
pub struct Utf8ValuesView<'a>(Utf8StorageView<'a>);

/// The borrowed form of [`Utf8Storage`], with the same invariants.
///
/// Row access branches on the storage once per row. The storage is the same for every row of a
/// batch, so the branch is easy to predict.
#[derive(Clone, Copy)]
enum Utf8StorageView<'a> {
    Views {
        views: &'a [BinaryView],
        buffers: &'a [ByteBuffer],
    },
    Offsets {
        offsets: &'a [u32],
        bytes: &'a [u8],
    },
}

// The row accessors below are not generic, so they carry `#[inline]`. Without it, a row loop that
// another crate instantiates cannot inline them and makes one call per row. The
// `row_fn_utf8_input` benchmark measures this.
impl Utf8Values {
    #[inline]
    fn view(&self) -> Utf8ValuesView<'_> {
        Utf8ValuesView(match &self.0 {
            Utf8Storage::Views { views, buffers } => Utf8StorageView::Views {
                views: views.as_slice(),
                buffers,
            },
            Utf8Storage::Offsets { offsets, bytes } => Utf8StorageView::Offsets {
                offsets: offsets.as_slice(),
                bytes: bytes.as_slice(),
            },
        })
    }
}

impl ViewLen for Utf8ValuesView<'_> {
    #[inline]
    fn len(&self) -> usize {
        match self.0 {
            Utf8StorageView::Views { views, .. } => views.len(),
            Utf8StorageView::Offsets { offsets, .. } => offsets.len() - 1,
        }
    }
}

impl<'a> Utf8ValuesView<'a> {
    #[inline]
    fn value(&self, index: usize) -> &'a str {
        assert!(
            index < self.len(),
            "row index must be less than {}, got {index}",
            self.len()
        );

        // SAFETY: the assertion bounds `index` by the row count.
        unsafe { self.value_unchecked(index) }
    }

    /// Reads one row without checking its index.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`ViewLen::len`].
    #[inline]
    unsafe fn value_unchecked(&self, index: usize) -> &'a str {
        match self.0 {
            Utf8StorageView::Views { views, buffers } => {
                // SAFETY: forwarded from this method's contract.
                let view = unsafe { views.get_unchecked(index) };
                view_str(view, buffers)
            }
            Utf8StorageView::Offsets { offsets, bytes } => {
                // SAFETY: the caller bounds `index` by the row count, and there is one more offset
                // than rows.
                let start = unsafe { *offsets.get_unchecked(index) } as usize;
                // SAFETY: the same bound puts `index + 1` at or before the terminal offset.
                let end = unsafe { *offsets.get_unchecked(index + 1) } as usize;

                // SAFETY: `offsets_tile_utf8` proved in `decode_utf8` that the offsets never
                // decrease and end within `bytes`.
                let value = unsafe { bytes.get_unchecked(start..end) };

                // SAFETY: `offsets_tile_utf8` also proved that every row is valid UTF-8.
                unsafe { std::str::from_utf8_unchecked(value) }
            }
        }
    }
}

/// Returns the string that a validated view addresses.
#[inline]
fn view_str<'a>(view: &'a BinaryView, buffers: &'a [ByteBuffer]) -> &'a str {
    let bytes = if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view = view.as_view();
        &buffers[view.buffer_index as usize][view.as_range()]
    };

    // SAFETY: `decode_views` validated every view and replaced each null view with an empty one.
    unsafe { std::str::from_utf8_unchecked(bytes) }
}

/// Decodes `array` into a column whose every row addresses valid UTF-8.
fn decode_utf8(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Utf8Values> {
    // A `VarBin` column keeps its offsets, so decoding builds no 16-byte view per row. Offsets that
    // do not fit in `u32`, or that delimit invalid UTF-8 at any row, fall back to views. The view
    // path reports invalid UTF-8 at a valid row and replaces each null row with an empty string.
    if let Some(varbin) = array.as_opt::<VarBin>() {
        let offsets = varbin.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let bytes = varbin.bytes_handle().try_to_host_sync()?;

        if let Some(offsets) = to_u32_offsets(offsets)
            && offsets_tile_utf8(&offsets, &bytes)
        {
            return Ok(Utf8Values(Utf8Storage::Offsets { offsets, bytes }));
        }
    }

    decode_views(array, ctx)
}

/// Returns `offsets` as `u32`, or `None` when an offset does not fit.
///
/// 32-bit offsets are reinterpreted without a copy. A negative `i32` offset becomes a large `u32`
/// that [`offsets_tile_utf8`] rejects.
fn to_u32_offsets(offsets: PrimitiveArray) -> Option<Buffer<u32>> {
    match offsets.ptype() {
        PType::U32 => Some(offsets.to_buffer::<u32>()),
        PType::I32 => Some(offsets.reinterpret_cast(PType::U32).to_buffer::<u32>()),
        ptype => match_each_integer_ptype!(ptype, |P| { copy_to_u32(offsets.as_slice::<P>()) }),
    }
}

/// Copies `offsets` into a new `u32` buffer, or returns `None` when an offset does not fit.
fn copy_to_u32<P: Copy + TryInto<u32>>(offsets: &[P]) -> Option<Buffer<u32>> {
    offsets
        .iter()
        .map(|&offset| offset.try_into().ok())
        .collect()
}

/// Decodes `array` into views and data buffers whose every view addresses valid UTF-8.
///
/// Validation also replaces each null view with an empty one. Canonical arrays built from
/// buffer handles do not carry host-content validation evidence, so this step remains necessary
/// before unchecked string access, even when an earlier decode validated the same input.
fn decode_views(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Utf8Values> {
    let array = array.execute::<VarBinViewArray>(ctx)?;
    let views = Buffer::<BinaryView>::from_byte_buffer(array.views_handle().try_to_host_sync()?);
    let buffers: Arc<[ByteBuffer]> = Arc::from(
        array
            .data_buffers()
            .iter()
            .map(BufferHandle::try_to_host_sync)
            .collect::<VortexResult<Vec<_>>>()?,
    );

    let validity = array.varbinview_validity();
    let views = VarBinViewData::validate_and_fix(views, &buffers, array.dtype(), &validity, ctx)?;

    Ok(Utf8Values(Utf8Storage::Views { views, buffers }))
}

/// Extracts the one string held by a UTF-8 batch constant.
fn decode_constant_utf8(array: &ArrayRef) -> VortexResult<BufferString> {
    let Some(constant) = array.as_opt::<Constant>() else {
        vortex_bail!(
            "a Utf8 batch constant must use the Constant encoding, got {}",
            array.encoding_id()
        );
    };
    let scalar = constant.scalar();
    let Some(ScalarValue::Utf8(value)) = scalar.value() else {
        vortex_bail!("a Utf8 batch constant must contain a non-null value, got {scalar}");
    };

    Ok(value.clone())
}

// SAFETY: decoding proves that every row of either storage addresses valid UTF-8 inside buffers
// the column owns. View storage replaces null rows with empty views, and offset storage validates
// null rows like valid ones. The borrowed view reports that exact stable length. Null rows are
// therefore safe for dense callbacks whose outputs the executor masks with the input validity.
unsafe impl InputElement for Utf8Column {
    type Column = Utf8Values;
    type Constant = BufferString;
    type View<'a> = Utf8ValuesView<'a>;
    type Elem<'a> = &'a str;

    const DENSE_SAFE: bool = true;
    const DECODE_INFALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Utf8(_)),
            "expected a Utf8 column, got {dtype}"
        );

        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        decode_utf8(array, ctx)
    }

    fn decode_constant(array: ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<Self::Constant> {
        decode_constant_utf8(&array)
    }

    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }

    #[inline]
    fn get(column: &Self::Column, index: usize) -> &str {
        column.view().value(index)
    }

    #[inline]
    fn get_constant(constant: &Self::Constant) -> &str {
        constant.as_str()
    }

    #[inline]
    fn view(column: &Self::Column) -> Self::View<'_> {
        column.view()
    }

    #[inline]
    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> &'a str
    where
        Self: 'a,
    {
        view.value(index)
    }

    #[inline]
    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> &'a str
    where
        Self: 'a,
    {
        // SAFETY: the executor validated `index` against this view's exact length.
        unsafe { view.value_unchecked(index) }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use super::Utf8Column;
    use super::Utf8Storage;
    use crate::ArrayRef;
    use crate::IntoArray as _;
    use crate::VortexSessionExecute as _;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::primitive::PrimitiveArrayExt as _;
    use crate::arrays::varbinview::BinaryView;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::VecExecutionArgs;
    use crate::scalar_fn::unstable::row::InputElement;
    use crate::scalar_fn::unstable::row::RowFn;
    use crate::scalar_fn::unstable::row::RowVisitor;
    use crate::scalar_fn::unstable::row::execute_rows;
    use crate::validity::Validity;

    #[derive(Clone)]
    struct StartsWithPre;

    impl RowFn for StartsWithPre {
        type Options = EmptyOptions;
        const ARG_NAMES: &'static [&'static str] = &["value"];
        const INFALLIBLE: bool = true;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("test.starts_with_pre");
            *ID
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &EmptyOptions,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::VisitResult> {
            visitor.visit::<(Utf8Column,), bool>(|(value,)| value.starts_with("pre"))
        }
    }

    const NULLABLE_VALUES: [Option<&str>; 4] = [
        Some("prefix"), // Matches.
        None,           // Null row.
        Some("héllo"),  // Multibyte, does not match.
        Some("pre"),    // Matches exactly.
    ];

    #[rstest]
    #[case::offsets(
        VarBinArray::from_iter(NULLABLE_VALUES, DType::Utf8(Nullability::Nullable)).into_array()
    )]
    #[case::views(VarBinViewArray::from_iter_nullable_str(NULLABLE_VALUES).into_array())]
    fn row_fn_reads_each_storage_and_propagates_nulls(#[case] input: ArrayRef) -> VortexResult<()> {
        let args = VecExecutionArgs::new(vec![input.clone()], input.len());
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let output = execute_rows(&StartsWithPre, &EmptyOptions, &args, &mut ctx)?;

        let expected = BoolArray::from_iter([
            Some(true),  //
            None,        //
            Some(false), //
            Some(true),  //
        ]);
        assert_arrays_eq!(output, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn input_reads_inline_and_referenced_views() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(["short", "a referenced string"]);
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let column = Utf8Column::decode(array.into_array(), &mut ctx)?;

        assert!(matches!(column.0, Utf8Storage::Views { .. }));
        assert_eq!(Utf8Column::get(&column, 0), "short");
        assert_eq!(Utf8Column::get(&column, 1), "a referenced string");

        Ok(())
    }

    /// Offsets that fit in `u32` are read in place, and 32-bit offsets are not copied.
    #[rstest]
    #[case::u32(PrimitiveArray::from_iter([0u32, 0, 6, 7]), true)]
    #[case::i32(PrimitiveArray::from_iter([0i32, 0, 6, 7]), true)]
    #[case::u8(PrimitiveArray::from_iter([0u8, 0, 6, 7]), false)]
    #[case::i64(PrimitiveArray::from_iter([0i64, 0, 6, 7]), false)]
    fn varbin_input_reads_offsets(
        #[case] offsets: PrimitiveArray,
        #[case] shares_offsets: bool,
    ) -> VortexResult<()> {
        let bytes = ByteBuffer::from("héllo!".as_bytes().to_vec());
        let array = VarBinArray::try_new(
            offsets.clone().into_array(),
            bytes.clone(),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        )?;
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let column = Utf8Column::decode(array.into_array(), &mut ctx)?;

        let Utf8Storage::Offsets {
            offsets: decoded_offsets,
            bytes: decoded_bytes,
        } = &column.0
        else {
            vortex_bail!("expected offset storage");
        };
        assert_eq!(decoded_bytes.as_ptr(), bytes.as_ptr());
        assert_eq!(
            decoded_offsets.as_ptr().cast::<u8>() == offsets.buffer_handle().as_host().as_ptr(),
            shares_offsets
        );
        for (index, expected) in ["", "héllo", "!"].into_iter().enumerate() {
            assert_eq!(Utf8Column::get(&column, index), expected);
        }

        Ok(())
    }

    #[test]
    fn varbin_input_reads_sliced_offsets() -> VortexResult<()> {
        let array = VarBinArray::from_iter_nonnull(
            ["before", "héllo", "a longer string", "after"],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array()
        .slice(1..3)?;
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let column = Utf8Column::decode(array, &mut ctx)?;

        assert!(matches!(column.0, Utf8Storage::Offsets { .. }));
        assert_eq!(Utf8Column::get(&column, 0), "héllo");
        assert_eq!(Utf8Column::get(&column, 1), "a longer string");

        Ok(())
    }

    /// Offset storage hands null rows to dense callbacks, so invalid bytes at a null row select
    /// view storage, which replaces null rows with empty strings.
    #[test]
    fn varbin_input_with_invalid_null_bytes_reads_views() -> VortexResult<()> {
        let array = VarBinArray::try_new(
            PrimitiveArray::from_iter([0u32, 1, 2]).into_array(),
            ByteBuffer::from(vec![b'a', 0xff]),
            DType::Utf8(Nullability::Nullable),
            Validity::from_iter([true, false]),
        )?;
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let column = Utf8Column::decode(array.into_array(), &mut ctx)?;

        assert!(matches!(column.0, Utf8Storage::Views { .. }));
        assert_eq!(Utf8Column::get(&column, 0), "a");
        assert_eq!(Utf8Column::get(&column, 1), "");

        Ok(())
    }

    #[test]
    fn input_sanitizes_unvalidated_null_views() -> VortexResult<()> {
        let invalid_view = BinaryView::from(u128::MAX);
        let views = BufferHandle::new_host(Buffer::from(vec![invalid_view]).into_byte_buffer());
        let validity = BoolArray::from_iter([false]).into_array();
        let array = VarBinViewArray::new_handle(
            views,
            Default::default(),
            DType::Utf8(Nullability::Nullable),
            Validity::Array(validity),
        );
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let column = Utf8Column::decode(array.into_array(), &mut ctx)?;

        assert_eq!(Utf8Column::get(&column, 0), "");

        Ok(())
    }

    #[rstest]
    #[case::empty(0)]
    #[case::one(1)]
    #[case::word_remainder(65)]
    fn input_sanitizes_sliced_null_views_repeatedly(#[case] len: usize) -> VortexResult<()> {
        let views: Buffer<BinaryView> = (0..len + 2)
            .map(|index| {
                if index % 2 == 0 {
                    BinaryView::from(u128::MAX)
                } else {
                    BinaryView::make_view(b"valid", 0, 0)
                }
            })
            .collect();
        let validity = BoolArray::from_iter((0..len + 2).map(|index| index % 2 != 0)).into_array();
        let array = VarBinViewArray::new_handle(
            BufferHandle::new_host(views.into_byte_buffer()),
            Default::default(),
            DType::Utf8(Nullability::Nullable),
            Validity::Array(validity),
        )
        .into_array()
        .slice(1..len + 1)?;
        let mut ctx = VortexSession::empty().create_execution_ctx();
        for _ in 0..2 {
            let column = Utf8Column::decode(array.clone(), &mut ctx)?;
            for index in 0..len {
                assert_eq!(
                    Utf8Column::get(&column, index),
                    if index % 2 == 0 { "valid" } else { "" }
                );
            }
        }
        Ok(())
    }

    #[test]
    fn input_rejects_unvalidated_utf8() {
        let invalid_view = BinaryView::make_view(&[0xff], 0, 0);
        let views = BufferHandle::new_host(Buffer::from(vec![invalid_view]).into_byte_buffer());
        let array = VarBinViewArray::new_handle(
            views,
            Default::default(),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        );
        let mut ctx = VortexSession::empty().create_execution_ctx();

        assert!(Utf8Column::decode(array.into_array(), &mut ctx).is_err());
    }

    #[rstest]
    #[case("inlined")]
    #[case("a referenced batch constant")]
    fn constant_decodes_inlined_and_referenced_values(#[case] value: &str) -> VortexResult<()> {
        let array = ConstantArray::new(Scalar::from(value), 8).into_array();
        let mut ctx = VortexSession::empty().create_execution_ctx();

        let constant = Utf8Column::decode_constant(array, &mut ctx)?;

        assert_eq!(Utf8Column::get_constant(&constant), value);

        Ok(())
    }
}
