// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::varbinview::{VarBinViewArray, VarBinViewVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<VarBinViewVTable> for VarBinViewVTable {
    fn len(array: &VarBinViewArray) -> usize {
        array.views.len()
    }

    fn dtype(array: &VarBinViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &VarBinViewArray, state: &mut H) {
        array.dtype.hash(state);
        for buffer in array.buffers.iter() {
            buffer.array_hash(state);
        }
        array.views.array_hash(state);
        array.validity.array_hash(state);
    }

    fn array_eq(array: &VarBinViewArray, other: &VarBinViewArray) -> bool {
        array.dtype == other.dtype
            && array.buffers.len() == other.buffers.len()
            && array
                .buffers
                .iter()
                .zip(other.buffers.iter())
                .all(|(a, b)| a.array_eq(b))
            && array.views.array_eq(&other.views)
            && array.validity.array_eq(&other.validity)
    }
}
