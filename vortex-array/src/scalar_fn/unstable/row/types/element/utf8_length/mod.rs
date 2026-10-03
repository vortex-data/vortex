// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 byte lengths read from native metadata without string preparation.
//!
//! [`Utf8OffsetLengthColumn`] and [`Utf8ViewLengthColumn`] read lengths without acquiring payload
//! buffers or constructing strings. The caller selects the physical layout before row execution.
//! Null-slot lengths are unspecified, and the executor preserves strict null propagation.

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

/// A UTF-8 input element that yields [`u64`] byte lengths from native [`u32`] offsets.
///
/// Non-constant, nonempty inputs **must** use [`VarBin`] with non-nullable [`u32`] offsets. Other
/// layouts are rejected even when their dtype is [`DType::Utf8`]. Empty UTF-8 arrays and _batch
/// constants_ (one value shared by every row) are supported separately.
///
/// Decoding validates offset bounds against payload length metadata without acquiring payload
/// buffers or constructing strings. Null-slot lengths are unspecified, and the executor propagates
/// validity. Layout selection must occur before dtype-only [RowFn dispatch].
///
/// [RowFn dispatch]: crate::scalar_fn::unstable::row::RowFn::dispatch
pub struct Utf8OffsetLengthColumn;

/// A UTF-8 input element that yields [`u64`] byte lengths from native string-view headers.
///
/// Non-constant, nonempty inputs **must** use [`VarBinView`]. Other layouts are rejected even when
/// their dtype is [`DType::Utf8`]. Empty UTF-8 arrays and batch constants are supported separately.
/// Decoding retains the headers without acquiring payload buffers or constructing strings, so
/// unreadable null payloads do not prevent header access.
///
/// See [`Utf8OffsetLengthColumn`] for the shared null-propagation and layout-selection contract.
pub struct Utf8ViewLengthColumn;

// SAFETY: OffsetMetadata::decode retains monotonic offsets with one terminal entry beyond the
// stable row domain. Length access only subtracts those offsets, so null payloads are never read.
// The executor propagates validity because null-slot lengths are unspecified.
unsafe impl InputElement for Utf8OffsetLengthColumn {
    type Column = OffsetMetadata;
    type Constant = u64;
    type View<'a> = &'a OffsetMetadata;
    type Elem<'a> = u64;

    const DENSE_SAFE: bool = true;

    // Other legal UTF-8 layouts require a different input element.
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

        // SAFETY: the caller bounds index by the row count, and decode retains rows + 1 offsets.
        let start = unsafe { *offsets.get_unchecked(index) };
        // SAFETY: the same row bound puts index + 1 within the retained terminal offset.
        let end = unsafe { *offsets.get_unchecked(index + 1) };

        u64::from(end - start)
    }
}

/// Owned string-view headers, without string payload buffers.
pub struct Utf8LengthViews {
    /// One initialized header per row, including null rows whose lengths are unspecified.
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

    // Other legal UTF-8 layouts require a different input element.
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
        // SAFETY: the caller bounds index by the exact retained view domain.
        let header = unsafe { view.views.as_slice().get_unchecked(index) };

        u64::from(header.len())
    }
}

#[cfg(test)]
mod tests;
