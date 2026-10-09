// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The sources and stages plans compile to.
//!
//! Every operator does work proportional to the rows it is handed: no operator looks at a batch
//! twice, holds more than the batches it must join, or scans inlets it does not read.

mod concat;
mod eval;
mod filter;
mod pack;
mod port;
mod scan;
mod take;

pub(crate) use concat::*;
pub(crate) use eval::*;
pub(crate) use filter::*;
pub(crate) use pack::*;
pub(crate) use port::*;
pub(crate) use scan::*;
pub(crate) use take::*;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::plan::pipeline::Inlet;

/// An inlet capacity that never blocks the writer, for a reader that needs its inlet whole.
const UNBOUNDED: usize = usize::MAX;

/// Takes the first `len` rows of the inlet's front batch: the batch itself when it is that
/// long, otherwise a slice, leaving the rest in place.
pub(crate) fn take_rows(inlet: &mut Inlet<'_>, len: usize) -> VortexResult<ArrayRef> {
    let front = inlet
        .peek_mut()
        .ok_or_else(|| vortex_err!("Taking rows from an empty inlet"))?;
    if front.len() == len {
        return inlet
            .take()
            .ok_or_else(|| vortex_err!("Taking rows from an empty inlet"));
    }
    let rest = front.slice(len..front.len())?;
    std::mem::replace(front, rest).slice(0..len)
}

/// Joins arrays covering consecutive rows into one.
pub(crate) fn join(dtype: &DType, mut arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    match arrays.len() {
        0 => Ok(Canonical::empty(dtype).into_array()),
        1 => Ok(arrays.remove(0)),
        _ => Ok(vortex_array::arrays::ChunkedArray::try_new(arrays, dtype.clone())?.into_array()),
    }
}

/// Takes every batch of a closed inlet.
pub(crate) fn drain(inlet: &mut Inlet<'_>) -> Vec<ArrayRef> {
    let mut batches = Vec::with_capacity(inlet.len());
    while let Some(batch) = inlet.take() {
        batches.push(batch);
    }
    batches
}
