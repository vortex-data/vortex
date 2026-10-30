// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The sources and stages plans compile to.
//!
//! Every source and stage does work proportional to the rows it is handed: none looks at a
//! batch twice, holds more than the batches it must join, or scans inlets it does not read.

mod concat;
mod eval;
mod filter;
mod once;
mod pack;
mod scan;
mod take;
mod whole;

pub(crate) use concat::*;
pub(crate) use eval::*;
pub(crate) use filter::*;
pub(crate) use once::*;
pub(crate) use pack::*;
pub(crate) use scan::*;
use smallvec::SmallVec;
pub(crate) use take::*;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
pub(crate) use whole::*;

use crate::plan::physical::operator::Cx;
use crate::plan::physical::pipeline::Inlet;

/// An inlet capacity that never blocks the sink, for a source that needs its inlet whole.
pub(crate) const UNBOUNDED: usize = usize::MAX;

/// Takes the first `len` rows of the inlet, across as many batches as hold them, slicing the last
/// and leaving its rest in place. Rows spanning batches come back chunked, not copied.
///
/// The rest is optimized with `session` before it goes back into the inlet. A slice is lazy, so
/// without this, each take would wrap the rest in one more slice, and every later take would
/// optimize the full chain of slices again.
fn take_rows(
    inlet: &mut Inlet<'_>,
    len: usize,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let mut pieces: SmallVec<[ArrayRef; 2]> = SmallVec::new();
    let mut left = len;
    while left > 0 {
        let front = inlet
            .peek_mut()
            .ok_or_else(|| vortex_err!("Taking {len} rows from an inlet holding fewer"))?;
        if front.len() <= left {
            left -= front.len();
            pieces.extend(inlet.take());
        } else {
            let rest = front.slice(left..front.len())?.optimize(session)?;
            pieces.push(std::mem::replace(front, rest).slice(0..left)?);
            left = 0;
        }
    }
    if pieces.len() == 1 {
        return Ok(pieces.swap_remove(0));
    }
    let dtype = pieces[0].dtype().clone();
    // SAFETY: the pieces come from one inlet, whose batches share its sink's dtype.
    Ok(unsafe { ChunkedArray::new_unchecked(pieces, dtype) }.into_array())
}

/// Joins arrays covering consecutive rows into one.
pub(crate) fn join(dtype: &DType, mut arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    match arrays.len() {
        0 => Ok(Canonical::empty(dtype).into_array()),
        1 => Ok(arrays.remove(0)),
        _ => Ok(ChunkedArray::try_new(arrays, dtype.clone())?.into_array()),
    }
}

/// Takes every batch of a closed inlet.
fn drain(inlet: &mut Inlet<'_>) -> Vec<ArrayRef> {
    let mut batches = Vec::with_capacity(inlet.len());
    while let Some(batch) = inlet.take() {
        batches.push(batch);
    }
    batches
}

/// Joins the whole of closed inlet `index` into one array of `dtype`.
fn join_inlet(cx: &mut Cx<'_>, index: usize, dtype: &DType) -> VortexResult<ArrayRef> {
    join(dtype, drain(&mut cx.inlet(index)))
}
