// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::DynArray;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnRef;
use crate::stats::ArrayStats;

// ScalarFnArray has a variable number of slots (one per child)

#[derive(Clone, Debug)]
pub struct ScalarFnArray {
    pub(super) scalar_fn: ScalarFnRef,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) stats: ArrayStats,
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

        let slots = children.into_iter().map(Some).collect();

        Ok(Self {
            scalar_fn: bound,
            dtype,
            len,
            slots,
            stats: Default::default(),
        })
    }

    /// Get the scalar function bound to this array.
    #[allow(clippy::same_name_method)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }

    /// Get a child array by index.
    pub fn get_child(&self, idx: usize) -> &ArrayRef {
        self.slots[idx]
            .as_ref()
            .vortex_expect("ScalarFnArray child slot")
    }

    /// Get the number of children.
    pub fn nchildren(&self) -> usize {
        self.slots.len()
    }

    /// Iterate over the children arrays without allocation.
    pub fn iter_children(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        self.slots
            .iter()
            .map(|s| s.as_ref().vortex_expect("ScalarFnArray child slot"))
    }

    /// Get the children arrays of this scalar function array.
    pub fn children(&self) -> Vec<ArrayRef> {
        self.iter_children().cloned().collect()
    }
}
