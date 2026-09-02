// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 input columns for row functions.
//!
//! [`Utf8Column`] decodes a `Utf8` column once per batch and hands each row callback a
//! [`Utf8View`]. A view dereferences to `str`, so a callback can call `str` methods directly. It
//! also exposes the string-view metadata that a comparison kernel can use before it reads the
//! bytes.

use std::ops::Deref;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewArrayExt as _;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::scalar::ScalarValue;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::ViewLen;

/// A UTF-8 input element that yields [`Utf8View`] values.
pub struct Utf8Column;

/// One decoded UTF-8 column.
pub struct Utf8Values {
    views: Buffer<BinaryView>,
    buffers: Arc<[ByteBuffer]>,
}

impl Utf8Values {
    /// Build a single-value representation of one batch-constant string.
    fn single(value: ByteBuffer) -> Self {
        let view = BinaryView::make_view(value.as_slice(), 0, 0);
        let buffers: Vec<ByteBuffer> = if view.is_inlined() {
            Vec::new()
        } else {
            vec![value]
        };

        Self {
            views: Buffer::from(vec![view]),
            buffers: Arc::from(buffers),
        }
    }

    fn views(&self) -> &[BinaryView] {
        self.views.as_slice()
    }
}

/// A borrowed UTF-8 column prepared for a row loop.
#[derive(Clone, Copy)]
pub struct Utf8ValuesView<'a> {
    views: &'a [BinaryView],
    buffers: &'a [ByteBuffer],
}

impl ViewLen for Utf8ValuesView<'_> {
    fn len(&self) -> usize {
        self.views.len()
    }
}

/// One UTF-8 value together with its Vortex string-view representation.
#[derive(Clone, Copy)]
pub struct Utf8View<'a> {
    view: &'a BinaryView,
    buffers: &'a [ByteBuffer],
}

impl<'a> Utf8View<'a> {
    /// Return the raw Vortex string view.
    ///
    /// The view reports the byte length without a read of the value.
    pub fn raw_view(&self) -> &BinaryView {
        self.view
    }

    /// Return whether the complete string is stored inside the view.
    pub fn is_inlined(&self) -> bool {
        self.view.is_inlined()
    }

    /// Return the string's first four bytes, or the complete string when it is shorter.
    pub fn prefix(&self) -> &[u8] {
        if self.view.is_inlined() {
            let value = self.view.as_inlined().value();
            &value[..value.len().min(4)]
        } else {
            &self.view.as_view().prefix
        }
    }

    /// Return the complete UTF-8 string with the lifetime of the decoded column.
    ///
    /// [`Deref`] borrows from the view instead, so a value that must outlive this view uses this
    /// method.
    pub fn as_str(&self) -> &'a str {
        let bytes = if self.view.is_inlined() {
            self.view.as_inlined().value()
        } else {
            let view = self.view.as_view();
            &self.buffers[view.buffer_index as usize][view.as_range()]
        };

        // SAFETY: the `VarBinViewArray` Utf8 invariant requires every valid view to contain UTF-8.
        // `decode_utf8` replaces each null row's view before this value can be constructed.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }
}

impl Deref for Utf8View<'_> {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Utf8View<'_> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Decodes `array` into views and data buffers whose every view addresses valid UTF-8.
///
/// The second [`VarBinViewArray::try_new`] is the sanitization step, not a round trip: it
/// validates each valid view and replaces every null row's view with an empty one. The
/// [`InputElement`] implementation relies on that, so a dense callback can read a null row's
/// payload and [`Utf8View::as_str`] can skip the UTF-8 check. Removing it makes those unsafe.
fn decode_utf8(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Utf8Values> {
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
    let array = VarBinViewArray::try_new(
        views,
        Arc::clone(&buffers),
        array.dtype().clone(),
        validity,
        ctx,
    )?;
    let views = Buffer::<BinaryView>::from_byte_buffer(array.views_handle().try_to_host_sync()?);

    Ok(Utf8Values { views, buffers })
}

/// Extracts the one string held by a UTF-8 batch constant.
fn decode_constant_utf8(array: &ArrayRef) -> VortexResult<Utf8Values> {
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

    Ok(Utf8Values::single(value.inner().clone()))
}

fn utf8_view(values: &Utf8Values) -> Utf8ValuesView<'_> {
    Utf8ValuesView {
        views: values.views(),
        buffers: &values.buffers,
    }
}

fn value_at<'a>(view: &Utf8ValuesView<'a>, index: usize) -> Utf8View<'a> {
    Utf8View {
        view: &view.views[index],
        buffers: view.buffers,
    }
}

/// Read a UTF-8 view without checking its row index.
///
/// # Safety
///
/// `index` must be less than the length reported by `view`.
unsafe fn value_at_unchecked<'a>(view: &Utf8ValuesView<'a>, index: usize) -> Utf8View<'a> {
    // SAFETY: forwarded from this function's contract.
    let value = unsafe { view.views.get_unchecked(index) };

    Utf8View {
        view: value,
        buffers: view.buffers,
    }
}

// SAFETY: `decode_utf8` validates every valid view and replaces null rows with empty views. The
// borrowed view reports that exact stable length. Therefore each view addresses valid UTF-8, and
// null rows are safe for dense callbacks whose outputs the executor masks with the input validity.
unsafe impl InputElement for Utf8Column {
    type Column = Utf8Values;
    type Constant = Utf8Values;
    type View<'a> = Utf8ValuesView<'a>;
    type Elem<'a> = Utf8View<'a>;

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

    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_> {
        value_at(&utf8_view(column), index)
    }

    fn get_constant(constant: &Self::Constant) -> Self::Elem<'_> {
        value_at(&utf8_view(constant), 0)
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        utf8_view(column)
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a,
    {
        value_at(view, index)
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a,
    {
        // SAFETY: the executor validated `index` against this view's exact view slice length.
        unsafe { value_at_unchecked(view, index) }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Utf8Column;
    use crate::IntoArray as _;
    use crate::VortexSessionExecute as _;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::BinaryView;
    use crate::buffer::BufferHandle;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::unstable::row::InputElement;
    use crate::validity::Validity;

    #[test]
    fn input_reads_inline_referenced_and_view_metadata() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(["short", "a referenced string"]);
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let column = Utf8Column::decode(array.into_array(), &mut ctx)?;

        let inline = Utf8Column::get(&column, 0);
        assert_eq!(&*inline, "short");
        assert!(inline.is_inlined());
        assert_eq!(inline.prefix(), b"shor");

        let referenced = Utf8Column::get(&column, 1);
        assert_eq!(&*referenced, "a referenced string");
        assert!(!referenced.is_inlined());
        assert_eq!(referenced.prefix(), b"a re");
        assert_eq!(referenced.raw_view().len(), 19);

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

        assert_eq!(&*Utf8Column::get(&column, 0), "");

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
        let view = Utf8Column::get_constant(&constant);

        assert_eq!(&*view, value);
        assert_eq!(view.raw_view().len() as usize, value.len());

        Ok(())
    }
}
