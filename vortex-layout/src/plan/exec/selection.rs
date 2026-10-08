// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

/// A node's selection over `rows` of its plan domain, indexed by plan row.
#[derive(Clone, Debug)]
pub(crate) struct Selection {
    rows: Range<u64>,
    mask: Mask,
}

impl Selection {
    pub(crate) fn try_new(rows: Range<u64>, mask: Mask) -> VortexResult<Self> {
        vortex_ensure!(
            rows.start <= rows.end && mask.len() as u64 == rows.end - rows.start,
            "Mask of length {} does not cover rows {rows:?}",
            mask.len()
        );
        Ok(Self { rows, mask })
    }

    pub(crate) fn rows(&self) -> &Range<u64> {
        &self.rows
    }

    pub(crate) fn mask(&self) -> &Mask {
        &self.mask
    }

    /// The mask over `rows`, which must lie within this selection.
    pub(crate) fn slice(&self, rows: &Range<u64>) -> VortexResult<Mask> {
        let start = usize::try_from(rows.start - self.rows.start)?;
        let end = usize::try_from(rows.end - self.rows.start)?;
        Ok(self.mask.slice(start..end))
    }
}

/// Joins arrays covering consecutive rows into one. An empty list joins to an empty array.
pub(crate) fn join(dtype: &DType, mut arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    match arrays.len() {
        0 => Ok(Canonical::empty(dtype).into_array()),
        1 => Ok(arrays.remove(0)),
        _ => Ok(ChunkedArray::try_new(arrays, dtype.clone())?.into_array()),
    }
}
