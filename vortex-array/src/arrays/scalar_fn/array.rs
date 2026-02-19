// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::expr::ScalarFn;
use crate::stats::ArrayStats;

#[derive(Clone, Debug)]
pub struct ScalarFnArray {
    pub(super) scalar_fn: ScalarFn,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) children: Vec<ArrayRef>,
    pub(super) stats: ArrayStats,
}

impl ScalarFnArray {
    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(bound: ScalarFn, children: Vec<ArrayRef>, len: usize) -> VortexResult<Self> {
        let arg_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
        let dtype = bound.return_dtype(&arg_dtypes)?;

        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "ScalarFnArray must have children equal to the array length"
        );

        Ok(Self {
            scalar_fn: bound,
            dtype,
            len,
            children,
            stats: Default::default(),
        })
    }

    /// Get the scalar function bound to this array.
    #[allow(clippy::same_name_method)]
    pub fn scalar_fn(&self) -> &ScalarFn {
        &self.scalar_fn
    }

    /// Get the children arrays of this scalar function array.
    pub fn children(&self) -> &[ArrayRef] {
        &self.children
    }
}
