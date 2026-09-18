// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binding the codec's random access to Vortex: describing an array's buffers as a layout,
//! supplying the low bits out of the child array, and turning elements back into scalars.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::EliasFano;
use crate::array::EliasFanoSlotsView;
use crate::array::scalar_from_bits;
use crate::ef;
use crate::ef::LowBits;
use crate::malformed;

/// The low-bits child, as a source the codec can pull from.
///
/// Built per call rather than held: it borrows the execution context.
pub(crate) struct LowerSource<'a, 'c> {
    pub(crate) lower: &'a ArrayRef,
    pub(crate) ctx: &'c mut ExecutionCtx,
}

impl LowBits for LowerSource<'_, '_> {
    type Error = VortexError;

    /// One low part, through the child's own `scalar_at`, so a rewritten slot needs no case here.
    fn get(&mut self, rank: u64) -> VortexResult<u64> {
        self.lower
            .execute_scalar(usize::try_from(rank)?, self.ctx)?
            .as_primitive()
            .typed_value::<u64>()
            .ok_or_else(|| vortex_err!("Elias-Fano low-bits child holds no value at rank {rank}"))
    }
}

/// Carry a read's failure into Vortex's error type. A function rather than a `From` impl, for the
/// reason [`crate::malformed`] gives.
pub(crate) fn read_error(error: ef::ReadError<VortexError>) -> VortexError {
    match error {
        ef::ReadError::Malformed(error) => malformed(error),
        ef::ReadError::LowBits(error) => error,
    }
}

/// The layout an array describes, borrowed from its buffers.
///
/// The upper array is taken as raw bytes rather than a `BitBuffer`, which would strip alignment and
/// can reallocate to carry the three numbers the codec wants.
pub(crate) fn layout<'a>(array: ArrayView<'a, EliasFano>) -> VortexResult<ef::Layout<'a>> {
    let data = array.data();
    let (_, samples1) = data.sample_bytes()?;
    let upper = ef::Bits::new(
        data.upper_buffer().as_slice(),
        0,
        usize::try_from(data.upper_len())?,
    );
    Ok(ef::Layout::new(
        upper,
        samples1,
        data.lower_width(),
        data.first_rank(),
        array.len(),
    ))
}

/// The value at logical `index`.
pub(crate) fn access_at(
    array: ArrayView<'_, EliasFano>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let len = array.len();
    if index >= len {
        vortex_bail!(OutOfBounds: index, 0usize, len);
    }
    let reference_bits = array.data().reference_bits();
    let layout = layout(array)?;

    // The slots view borrows the array behind the `ArrayView`; the `lower()` accessor would borrow
    // the (`Copy`, stack-local) view itself.
    let lower = EliasFanoSlotsView::from_slots(array.slots()).lower;
    let mut source = LowerSource { lower, ctx };

    let element = ef::element_at(layout, index, &mut source).map_err(read_error)?;
    scalar_from_bits(array.dtype(), reference_bits.wrapping_add(element))
}
