// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::TypedArrayRef;
use crate::arrays::ScalarFnVTable;
use crate::scalar_fn::ScalarFnRef;

// ScalarFnArray has a variable number of slots (one per child)

#[derive(Clone, Debug)]
pub struct ScalarFnData {
    pub(super) scalar_fn: ScalarFnRef,
}

impl ScalarFnData {
    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "ScalarFnArray must have children equal to the array length"
        );
        Ok(Self { scalar_fn })
    }

    /// Get the scalar function bound to this array.
    #[allow(clippy::same_name_method)]
    #[inline(always)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }
}

pub trait ScalarFnArrayExt: TypedArrayRef<ScalarFnVTable> {
    fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }

    fn child_at(&self, idx: usize) -> &ArrayRef {
        self.as_ref().slots()[idx]
            .as_ref()
            .vortex_expect("ScalarFnArray child slot")
    }

    fn child_count(&self) -> usize {
        self.as_ref().slots().len()
    }

    fn iter_children(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        (0..self.child_count()).map(|idx| self.child_at(idx))
    }

    fn children(&self) -> Vec<ArrayRef> {
        self.iter_children().cloned().collect()
    }
}
impl<T: TypedArrayRef<ScalarFnVTable>> ScalarFnArrayExt for T {}

impl Array<ScalarFnVTable> {
    /// Get the scalar function bound to this array.
    #[allow(clippy::same_name_method)]
    #[inline(always)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        ScalarFnArrayExt::scalar_fn(self)
    }

    /// Get the children arrays of this scalar function array.
    #[allow(clippy::same_name_method)]
    pub fn children(&self) -> Vec<ArrayRef> {
        ScalarFnArrayExt::children(self)
    }

    #[allow(clippy::same_name_method)]
    pub fn get_child(&self, idx: usize) -> &ArrayRef {
        ScalarFnArrayExt::child_at(self, idx)
    }

    #[allow(clippy::same_name_method)]
    pub fn nchildren(&self) -> usize {
        ScalarFnArrayExt::child_count(self)
    }

    #[allow(clippy::same_name_method)]
    pub fn iter_children(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        ScalarFnArrayExt::iter_children(self)
    }

    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        let arg_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;
        let data = ScalarFnData::try_new(scalar_fn.clone(), children.clone(), len)?;
        let vtable = ScalarFnVTable { scalar_fn };
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(vtable, dtype, len, data)
                    .with_slots(children.into_iter().map(Some).collect()),
            )
        })
    }
}

impl ArrayView<'_, ScalarFnVTable> {
    #[allow(clippy::same_name_method)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        ScalarFnArrayExt::scalar_fn(self)
    }

    #[allow(clippy::same_name_method)]
    pub fn children(&self) -> Vec<ArrayRef> {
        ScalarFnArrayExt::children(self)
    }

    #[allow(clippy::same_name_method)]
    pub fn get_child(&self, idx: usize) -> &ArrayRef {
        ScalarFnArrayExt::child_at(self, idx)
    }

    #[allow(clippy::same_name_method)]
    pub fn nchildren(&self) -> usize {
        ScalarFnArrayExt::child_count(self)
    }

    #[allow(clippy::same_name_method)]
    pub fn iter_children(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        ScalarFnArrayExt::iter_children(self)
    }
}
