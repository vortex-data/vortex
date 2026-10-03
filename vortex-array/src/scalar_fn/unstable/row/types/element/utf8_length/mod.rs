// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 byte lengths read from native metadata without string preparation.
//!
//! These input elements yield `u64` lengths without acquiring payload buffers or validating UTF-8.
//! They retain initialized metadata for null slots. Their lengths are unspecified in those slots,
//! and the executor preserves strict null propagation.

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::Utf8Column;
use super::utf8_offset::OffsetMetadata;
use super::utf8_offset::constant_string;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::ViewLen;

/// A UTF-8 input element that yields byte lengths from native `u32` offsets.
///
/// Decode accepts VarBin inputs with non-nullable `u32` offsets, plus empty Utf8 arrays and batch
/// constants. Other nonempty physical layouts are rejected. The offset domain is validated using
/// payload length metadata, without acquiring payload buffers or producing strings.
///
/// Lengths in null slots are unspecified. Callbacks must rely on the executor for null propagation.
/// Physical input selection must occur outside dtype-only RowFn dispatch.
pub struct Utf8OffsetLengthColumn;

/// A UTF-8 input element that yields byte lengths from native string-view headers.
///
/// Decode accepts VarBinView inputs, plus empty Utf8 arrays and batch constants. Other nonempty
/// physical layouts are rejected. It retains the view headers without acquiring payload buffers or
/// producing strings. Header reads remain safe even when a null view has an unreadable payload.
///
/// Lengths in null slots are unspecified. Callbacks must rely on the executor for null propagation.
/// Physical input selection must occur outside dtype-only RowFn dispatch.
pub struct Utf8ViewLengthColumn;

// SAFETY: decode validates the metadata domain. Length access never borrows string payloads, and
// null lengths are unspecified. The executor propagates validity. The view retains the offsets.
unsafe impl InputElement for Utf8OffsetLengthColumn {
    type Column = OffsetMetadata;
    type Constant = u64;
    type View<'a> = &'a OffsetMetadata;
    type Elem<'a> = u64;

    const DENSE_SAFE: bool = true;

    // Other legal Utf8 layouts require a different input element.
    const DECODE_INFALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        Utf8Column::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        OffsetMetadata::decode(&array, ctx)
    }

    fn decode_constant(array: ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<u64> {
        Ok(constant_string(array)?.len() as u64)
    }

    fn can_decode_null_tolerant(array: &ArrayRef) -> VortexResult<bool> {
        Ok(array.is_empty() || array.is::<VarBin>())
    }

    fn get(column: &Self::Column, index: usize) -> u64 {
        u64::from(column.offsets[index + 1] - column.offsets[index])
    }

    fn get_constant(constant: &u64) -> u64 {
        *constant
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> u64
    where
        Self: 'a,
    {
        Self::get(view, index)
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> u64
    where
        Self: 'a,
    {
        let offsets = view.offsets.as_slice();
        // SAFETY: the executor bounds index by rows, and decode retains rows + 1 offsets.
        let start = unsafe { *offsets.get_unchecked(index) };
        let end = unsafe { *offsets.get_unchecked(index + 1) };
        u64::from(end - start)
    }
}

/// Owned string-view headers, without string payload buffers.
pub struct Utf8LengthViews {
    views: Buffer<BinaryView>,
}

impl ViewLen for &Utf8LengthViews {
    fn len(&self) -> usize {
        self.views.len()
    }
}

// SAFETY: one owned view corresponds to each row. Every length header is an initialized integer,
// including null slots. Null lengths are unspecified. The executor propagates validity. No bytes
// or unchecked strings are exposed, so UTF-8 validation is unnecessary.
unsafe impl InputElement for Utf8ViewLengthColumn {
    type Column = Utf8LengthViews;
    type Constant = u64;
    type View<'a> = &'a Utf8LengthViews;
    type Elem<'a> = u64;

    const DENSE_SAFE: bool = true;

    // Other legal Utf8 layouts require a different input element.
    const DECODE_INFALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        Utf8Column::validate(dtype)
    }

    fn decode(array: ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        Utf8Column::validate(array.dtype())?;
        if array.is_empty() {
            return Ok(Utf8LengthViews {
                views: Buffer::empty(),
            });
        }
        let array = array.as_opt::<VarBinView>().ok_or_else(|| {
            vortex_err!(
                "expected VarBinView for a Utf8 view-length column, got {}",
                array.encoding_id()
            )
        })?;
        let views = Buffer::from_byte_buffer(array.views_handle().try_to_host_sync()?);

        Ok(Utf8LengthViews { views })
    }

    fn decode_constant(array: ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<u64> {
        Ok(constant_string(array)?.len() as u64)
    }

    fn can_decode_null_tolerant(array: &ArrayRef) -> VortexResult<bool> {
        Ok(array.is_empty() || array.is::<VarBinView>())
    }

    fn get(column: &Self::Column, index: usize) -> u64 {
        u64::from(column.views[index].len())
    }

    fn get_constant(constant: &u64) -> u64 {
        *constant
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> u64
    where
        Self: 'a,
    {
        Self::get(view, index)
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> u64
    where
        Self: 'a,
    {
        // SAFETY: the executor bounds index by the exact retained view domain.
        u64::from(unsafe { view.views.as_slice().get_unchecked(index) }.len())
    }
}

#[cfg(test)]
mod tests;
