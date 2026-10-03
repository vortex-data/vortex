// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 row input borrowed directly from native offset storage.
//!
//! This input avoids constructing string views. It owns validated offsets and payload buffers,
//! validates UTF-8 on readable rows, and substitutes empty strings for null rows.

use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::Utf8Column;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::varbin::VarBinArrayExt as _;
use crate::arrays::varbin::VarBinArraySlotsExt as _;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::ViewLen;

/// A UTF-8 input element that yields `&str` from native `u32` offset storage.
///
/// Callers must select this element for VarBin inputs with non-nullable `u32` offsets. Decode
/// rejects other nonempty physical layouts even when their logical dtype is Utf8. Batch constants
/// and empty Utf8 arrays are supported separately. [`super::Utf8Column`] remains the general input
/// element for other encodings.
///
/// Decode preserves offset and payload owners and validates each readable UTF-8 range. Null rows
/// yield empty strings, while the executor propagates their validity. This type does not select a
/// physical representation through the dtype-only [`crate::scalar_fn::unstable::row::RowFn::dispatch`].
pub struct Utf8OffsetColumn;

/// Owned, validated offsets and their readable row domain.
pub struct OffsetMetadata {
    pub(super) offsets: Buffer<u32>,
    readable: Mask,
}

impl OffsetMetadata {
    /// Validate raw metadata before it becomes a decoded column.
    fn from_parts(
        offsets: Buffer<u32>,
        readable: Mask,
        payload_len: usize,
        rows: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offsets.len() == rows + 1,
            "offset count must equal rows plus one, got {}",
            offsets.len()
        );
        vortex_ensure!(
            readable.len() == rows,
            "readable mask must match rows, got {}",
            readable.len()
        );
        vortex_ensure!(
            offsets.iter().all(|&offset| offset as usize <= payload_len),
            "offsets must be within {payload_len} payload bytes, got maximum offset {:?}",
            offsets.iter().max()
        );
        vortex_ensure!(
            offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "offsets must be monotonic, got decreasing pair {:?}",
            offsets.windows(2).find(|pair| pair[0] > pair[1])
        );

        Ok(Self { offsets, readable })
    }

    pub(super) fn decode(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Utf8Column::validate(array.dtype())?;
        if array.is_empty() {
            return Ok(Self {
                offsets: Buffer::from(vec![0]),
                readable: Mask::new_true(0),
            });
        }
        let array = array.as_opt::<VarBin>().ok_or_else(|| {
            vortex_err!(
                "expected VarBin for a Utf8 offset column, got {}",
                array.encoding_id()
            )
        })?;
        let primitive = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        vortex_ensure!(
            !primitive.dtype().is_nullable(),
            "expected non-nullable offsets, got {}",
            primitive.dtype()
        );
        vortex_ensure!(
            primitive.ptype() == PType::U32,
            "expected u32 offsets, got {}",
            primitive.dtype()
        );
        let offsets = primitive.to_buffer::<u32>();
        let payload_len = array.bytes_handle().len();
        let readable = array.varbin_validity().execute_mask(array.len(), ctx)?;

        Self::from_parts(offsets, readable, payload_len, array.len())
    }

    fn range(&self, index: usize) -> std::ops::Range<usize> {
        self.offsets[index] as usize..self.offsets[index + 1] as usize
    }
}

impl ViewLen for &OffsetMetadata {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

/// Owned offset storage whose readable ranges contain validated UTF-8.
pub struct Utf8OffsetValues {
    metadata: OffsetMetadata,
    bytes: ByteBuffer,
}

impl Utf8OffsetValues {
    /// Validate acquired bytes without constructing an array or borrowing an unchecked string.
    fn from_parts(metadata: OffsetMetadata, bytes: ByteBuffer) -> VortexResult<Self> {
        if metadata.readable.true_count() != 0 {
            vortex_ensure!(
                metadata.offsets[metadata.offsets.len() - 1] as usize <= bytes.len(),
                "offsets must fit the acquired payload, got end {} and payload length {}",
                metadata.offsets[metadata.offsets.len() - 1],
                bytes.len()
            );
        }
        for index in 0..metadata.offsets.len() - 1 {
            if metadata.readable.value(index) {
                let range = metadata.range(index);
                vortex_ensure!(
                    simdutf8::basic::from_utf8(&bytes[range]).is_ok(),
                    "expected valid UTF-8, got invalid bytes at row {index}"
                );
            }
        }

        Ok(Self { metadata, bytes })
    }

    fn value(&self, index: usize) -> &str {
        if !self.metadata.readable.value(index) {
            return "";
        }

        let bytes = &self.bytes[self.metadata.range(index)];
        // SAFETY: decode validated UTF-8 for every readable range and retains both buffers.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }
}

impl ViewLen for &Utf8OffsetValues {
    fn len(&self) -> usize {
        self.metadata.offsets.len() - 1
    }
}

pub(super) fn constant_string(array: ArrayRef) -> VortexResult<ByteBuffer> {
    Utf8Column::validate(array.dtype())?;
    let array = array
        .as_opt::<Constant>()
        .ok_or_else(|| vortex_err!("expected a Constant string, got {}", array.encoding_id()))?;
    let value = array
        .scalar()
        .as_utf8()
        .value()
        .ok_or_else(|| vortex_err!("expected a non-null constant, got {}", array.scalar()))?;

    Ok(value.inner().clone())
}

/// The owned, validated UTF-8 value of one batch constant.
///
/// Its private field prevents arbitrary byte buffers from reaching unchecked string access.
pub struct Utf8OffsetConstant(ByteBuffer);

// SAFETY: decode proves every readable range is bounded UTF-8. Null slots return empty strings.
// The borrowed column owns the offsets, readable mask, and bytes for its stable row domain.
unsafe impl InputElement for Utf8OffsetColumn {
    type Column = Utf8OffsetValues;
    type Constant = Utf8OffsetConstant;
    type View<'a> = &'a Utf8OffsetValues;
    type Elem<'a> = &'a str;

    const DENSE_SAFE: bool = true;

    // A legal Utf8 array can use a physical layout this element does not support.
    const DECODE_INFALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        Utf8Column::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let metadata = OffsetMetadata::decode(&array, ctx)?;
        let bytes = if metadata.readable.true_count() == 0 {
            ByteBuffer::empty()
        } else {
            array
                .as_opt::<VarBin>()
                .ok_or_else(|| vortex_err!("expected VarBin, got {}", array.encoding_id()))?
                .bytes_handle()
                .try_to_host_sync()?
        };
        Utf8OffsetValues::from_parts(metadata, bytes)
    }

    fn decode_constant(array: ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<Self::Constant> {
        Ok(Utf8OffsetConstant(constant_string(array)?))
    }

    fn can_decode_null_tolerant(array: &ArrayRef) -> VortexResult<bool> {
        Ok(array.is_empty() || array.is::<VarBin>())
    }

    fn get(column: &Self::Column, index: usize) -> &str {
        column.value(index)
    }

    fn get_constant(constant: &Self::Constant) -> &str {
        // SAFETY: a Utf8 scalar owns validated UTF-8 bytes.
        unsafe { std::str::from_utf8_unchecked(constant.0.as_slice()) }
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> &'a str {
        view.value(index)
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> &'a str {
        if !view.metadata.readable.value(index) {
            return "";
        }
        let offsets = view.metadata.offsets.as_slice();
        // SAFETY: the executor bounds index by rows, and decode retains rows + 1 offsets.
        let start = unsafe { *offsets.get_unchecked(index) } as usize;
        let end = unsafe { *offsets.get_unchecked(index + 1) } as usize;
        // SAFETY: decode validated both the payload range and UTF-8 for this readable row.
        let bytes = unsafe { view.bytes.as_slice().get_unchecked(start..end) };
        // SAFETY: from_parts validated UTF-8 for this readable range.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }
}

#[cfg(test)]
pub(super) mod tests;
