// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::plan::exec::Piece;

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
    pub(crate) fn slice(&self, rows: &Range<u64>) -> Mask {
        self.mask
            .slice(self.index(rows.start)..self.index(rows.end))
    }

    /// The number of selected rows in `rows`.
    pub(crate) fn count(&self, rows: &Range<u64>) -> usize {
        self.slice(rows).true_count()
    }

    fn index(&self, row: u64) -> usize {
        usize::try_from(row - self.rows.start).vortex_expect("selection index must fit in usize")
    }
}

/// A piece with no selected rows.
pub(crate) fn empty_piece(dtype: &DType, rows: Range<u64>) -> Piece {
    Piece {
        rows,
        array: Canonical::empty(dtype).into_array(),
    }
}

impl Piece {
    /// The part of this piece covering `sub`, which must lie within `self.rows`.
    pub(crate) fn slice_rows(&self, selection: &Selection, sub: Range<u64>) -> VortexResult<Piece> {
        if sub == self.rows {
            return Ok(self.clone());
        }
        let start = selection.count(&(self.rows.start..sub.start));
        let len = selection.count(&sub);
        Ok(Piece {
            rows: sub,
            array: self.array.slice(start..start + len)?,
        })
    }
}

/// Joins arrays covering consecutive rows.
pub(crate) fn join(dtype: &DType, mut arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    if arrays.len() == 1 {
        return Ok(arrays.remove(0));
    }
    Ok(ChunkedArray::try_new(arrays, dtype.clone())?.into_array())
}
