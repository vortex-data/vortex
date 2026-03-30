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
use crate::vtable::Array;

#[derive(Clone, Debug)]
pub struct ScalarFnData {
    pub(super) vtable: ScalarFnVTable,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) children: Vec<ArrayRef>,
    pub(super) stats: ArrayStats,
}

impl ScalarFnData {
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

    /// Returns the dtype of the array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

impl Array<ScalarFnVTable> {
    /// Get the scalar function bound to this array.
    #[allow(clippy::same_name_method)]
    #[inline(always)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        self.data().scalar_fn()
    }

    /// Get the children arrays of this scalar function array.
    #[allow(clippy::same_name_method)]
    pub fn children(&self) -> &[ArrayRef] {
        self.data().children()
    }

    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        Array::try_from_data(ScalarFnData::try_new(scalar_fn, children, len)?)
    }
}
