// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::Precision;
use crate::arrays::varbinview::{VarBinViewArray, VarBinViewVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<VarBinViewVTable> for VarBinViewVTable {
    fn len(array: &VarBinViewArray) -> usize {
        array.views.len()
    }

    fn dtype(array: &VarBinViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &VarBinViewArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        for buffer in array.buffers.iter() {
            buffer.array_hash(state, precision);
        }
        array.views.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &VarBinViewArray, other: &VarBinViewArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.buffers.len() == other.buffers.len()
            && array
                .buffers
                .iter()
                .zip(other.buffers.iter())
                .all(|(a, b)| a.array_eq(b, precision))
            && array.views.array_eq(&other.views, precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}
