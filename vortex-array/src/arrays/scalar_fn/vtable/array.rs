// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::Precision;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::dtype::DType;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<ScalarFnVTable> for ScalarFnVTable {
    fn len(array: &ScalarFnArray) -> usize {
        array.len
    }

    fn dtype(array: &ScalarFnArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ScalarFnArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &ScalarFnArray, state: &mut H, precision: Precision) {
        array.len.hash(state);
        array.dtype.hash(state);
        array.scalar_fn.hash(state);
        for child in &array.children {
            child.array_hash(state, precision);
        }
    }

    fn array_eq(array: &ScalarFnArray, other: &ScalarFnArray, precision: Precision) -> bool {
        if array.len != other.len {
            return false;
        }
        if array.dtype != other.dtype {
            return false;
        }
        if array.scalar_fn != other.scalar_fn {
            return false;
        }
        for (child, other_child) in array.children.iter().zip(other.children.iter()) {
            if !child.array_eq(other_child, precision) {
                return false;
            }
        }
        true
    }
}
