// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::functions::scalar::ScalarFn;
use crate::stats::ArrayStats;
use crate::vtable::ArrayVTable;
use crate::ArrayRef;
use vortex_dtype::DType;

#[derive(Clone, Debug)]
pub struct ScalarFnArray {
    // NOTE(ngates): we should fix vtables so we don't have to hold this
    pub(super) vtable: ArrayVTable,
    pub(super) scalar_fn: ScalarFn,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) children: Vec<ArrayRef>,
    pub(super) stats: ArrayStats,
}
