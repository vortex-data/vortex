// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::DynArray;
use crate::arrays::ScalarFnVTable;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnRef;
use crate::stats::ArrayStats;

#[derive(Clone, Debug)]
pub struct ScalarFnArray {
    pub(super) vtable: ScalarFnVTable,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) children: Vec<ArrayRef>,
    pub(super) stats: ArrayStats,
}

impl ScalarFnArray {
    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        let arg_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "ScalarFnArray must have children equal to the array length"
        );

        Ok(Self {
            vtable: ScalarFnVTable { scalar_fn },
            dtype,
            len,
            children,
            stats: Default::default(),
        })
    }

    /// Get the scalar function bound to this array.
    #[allow(clippy::same_name_method)]
    #[inline(always)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.vtable.scalar_fn
    }

    /// Get the children arrays of this scalar function array.
    pub fn children(&self) -> &[ArrayRef] {
        &self.children
    }
}
