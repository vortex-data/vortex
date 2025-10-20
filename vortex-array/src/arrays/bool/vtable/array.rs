// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::{BoolArray, BoolVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<BoolVTable> for BoolVTable {
    fn len(array: &BoolArray) -> usize {
        array.buffer.len()
    }

    fn dtype(array: &BoolArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &BoolArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &BoolArray, state: &mut H) {
        array.dtype.hash(state);
        // BooleanBuffer is from arrow-buffer, so we manually hash its components
        array.buffer.offset().hash(state);
        array.buffer.len().hash(state);
        array.buffer.inner().as_ptr().hash(state);
        array.validity.array_hash(state);
    }

    fn array_eq(array: &BoolArray, other: &BoolArray) -> bool {
        if array.dtype != other.dtype {
            return false;
        }
        // BooleanBuffer is from arrow-buffer, so we manually compare its components
        let buf1 = &array.buffer;
        let buf2 = &other.buffer;
        buf1.offset() == buf2.offset()
            && buf1.len() == buf2.len()
            && buf1.inner().as_ptr() == buf2.inner().as_ptr()
            && array.validity.array_eq(&other.validity)
    }
}
