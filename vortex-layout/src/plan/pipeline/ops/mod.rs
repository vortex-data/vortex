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
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::plan::pipeline::Inlet;

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
