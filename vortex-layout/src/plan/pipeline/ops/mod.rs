// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The sources and stages plans compile to.
//!
//! Every operator does work proportional to the rows it is handed: no operator looks at a batch
//! twice, holds more than the batches it must join, or scans inlets it does not read.

mod concat;
mod filter;
mod pack;
mod port;
mod scan;

pub(crate) use concat::*;
pub(crate) use filter::*;
pub(crate) use pack::*;
pub(crate) use port::*;
pub(crate) use scan::*;
use smallvec::SmallVec;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::plan::pipeline::Inlet;

/// Takes the first `len` rows of the inlet, across as many batches as hold them, slicing the last
/// and leaving its rest in place. Rows spanning batches come back chunked, not copied.
pub(crate) fn take_rows(inlet: &mut Inlet<'_>, len: usize) -> VortexResult<ArrayRef> {
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
            let rest = front.slice(left..front.len())?;
            pieces.push(std::mem::replace(front, rest).slice(0..left)?);
            left = 0;
        }
    }
    if pieces.len() == 1 {
        return Ok(pieces.swap_remove(0));
    }
    let dtype = pieces[0].dtype().clone();
    // SAFETY: the pieces come from one inlet, whose batches share its writer's dtype.
    Ok(unsafe { ChunkedArray::new_unchecked(pieces, dtype) }.into_array())
}
