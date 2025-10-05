// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::arrays::MaskedVTable;
use crate::arrays::masked::MaskedArray;
use crate::stats::ArrayStats;
use crate::vtable::OperationsVTable;
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<MaskedVTable> for MaskedVTable {
    fn slice(array: &MaskedArray, range: Range<usize>) -> ArrayRef {
        let child = array.child.slice(range.clone());
        let validity = array.validity.slice(range);

        MaskedArray {
            child,
            validity,
            dtype: array.dtype.clone(),
            stats: ArrayStats::default(),
        }
        .into_array()
    }

    fn scalar_at(array: &MaskedArray, index: usize) -> Scalar {
        // Invalid indices are handled by the entrypoint function.
        array.child.scalar_at(index).into_nullable()
    }
}
