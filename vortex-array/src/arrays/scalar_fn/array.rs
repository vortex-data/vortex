// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayCommon;
use crate::ArrayRef;
use crate::DynArray;
use crate::scalar_fn::ScalarFnRef;

#[derive(Clone, Debug)]
pub struct ScalarFnArray {
    pub(super) scalar_fn: ScalarFnRef,
    pub(super) common: ArrayCommon,
    pub(super) children: Vec<ArrayRef>,
}

impl ScalarFnArray {
    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(bound: ScalarFnRef, children: Vec<ArrayRef>, len: usize) -> VortexResult<Self> {
        let arg_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
        let dtype = bound.return_dtype(&arg_dtypes)?;

        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "ScalarFnArray must have children equal to the array length"
        );

        Ok(Self {
            scalar_fn: bound,
            common: ArrayCommon::new(len, dtype),
            children,
        })
    }

    /// Get the scalar function bound to this array.
    #[allow(clippy::same_name_method)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }

    /// Get the children arrays of this scalar function array.
    pub fn children(&self) -> &[ArrayRef] {
        &self.children
    }
}
