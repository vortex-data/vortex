// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ScalarFnVTable;
use crate::expr::functions::scalar::ScalarFn;
use crate::stats::ArrayStats;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;

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

impl ScalarFnArray {
    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(scalar_fn: ScalarFn, children: Vec<ArrayRef>, len: usize) -> VortexResult<Self> {
        let arg_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "ScalarFnArray must have children equal to the array length"
        );

        Ok(Self {
            vtable: ScalarFnVTable::new(scalar_fn.vtable().clone()).into_vtable(),
            scalar_fn,
            dtype,
            len,
            children,
            stats: Default::default(),
        })
    }
}
