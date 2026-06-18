// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::SmallVec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::ScalarFn;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnRef;

// ScalarFnArray has a variable number of slots (one per child)

#[derive(Clone, Debug)]
pub struct ScalarFnData {
    pub(super) scalar_fn: ScalarFnRef,
}

impl Display for ScalarFnData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "scalar_fn: {}", self.scalar_fn)
    }
}

impl ScalarFnData {
    /// Create a new ScalarFnArray from a scalar function and its children.
    fn build(scalar_fn: ScalarFnRef) -> Self {
        Self { scalar_fn }
    }

    /// Get the scalar function bound to this array.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }
}

pub trait ScalarFnArrayExt: TypedArrayRef<ScalarFn> {
    fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }

    fn child_at(&self, idx: usize) -> &ArrayRef {
        self.slots()[idx]
            .as_ref()
            .vortex_expect("ScalarFnArray child slot")
    }

    fn child_count(&self) -> usize {
        self.slots().len()
    }

    fn nchildren(&self) -> usize {
        self.child_count()
    }

    fn get_child(&self, idx: usize) -> &ArrayRef {
        self.child_at(idx)
    }

    fn iter_children(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        (0..self.child_count()).map(|idx| self.child_at(idx))
    }

    fn children(&self) -> Vec<ArrayRef> {
        self.iter_children().cloned().collect()
    }
}
impl<T: TypedArrayRef<ScalarFn>> ScalarFnArrayExt for T {}

impl Array<ScalarFn> {
    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<Self> {
        let slots: ArraySlots = children.into_iter().map(Some).collect();
        let len = Self::infer_len(&slots)?;
        Self::try_new_from_slots(scalar_fn, slots, len)
    }

    /// Create a new ScalarFnArray from a scalar function, children, and an explicit length.
    ///
    /// This is needed for zero-child scalar functions and deserialization paths where there is no
    /// child array to infer the length from.
    pub fn try_new_with_len(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        let slots: ArraySlots = children.into_iter().map(Some).collect();
        Self::try_new_from_slots(scalar_fn, slots, len)
    }

    /// Build the [`ArrayParts`] for a ScalarFnArray without materializing it.
    ///
    /// Mirrors [`try_new_with_len`](Self::try_new_with_len) but stops short of allocating the
    /// backing `ArrayRef`, so callers can drive the parts through [`ArrayParts::optimize`] and
    /// only pay the wrapper allocation when no reduction fires.
    #[inline]
    pub fn try_new_parts(
        scalar_fn: ScalarFnRef,
        children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<ArrayParts<ScalarFn>> {
        let slots: ArraySlots = children.into_iter().map(Some).collect();
        Self::try_new_parts_from_slots(scalar_fn, slots, len)
    }

    #[inline]
    fn try_new_from_slots(
        scalar_fn: ScalarFnRef,
        slots: ArraySlots,
        len: usize,
    ) -> VortexResult<Self> {
        let parts = Self::try_new_parts_from_slots(scalar_fn, slots, len)?;

        Ok(unsafe { Array::from_parts_unchecked(parts) })
    }

    #[inline]
    fn try_new_parts_from_slots(
        scalar_fn: ScalarFnRef,
        slots: ArraySlots,
        len: usize,
    ) -> VortexResult<ArrayParts<ScalarFn>> {
        Self::validate_arity(&scalar_fn, slots.len())?;
        Self::validate_children_len(&slots, len)?;

        let arg_dtypes = Self::arg_dtypes(&slots);
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        let vtable = ScalarFn { id: scalar_fn.id() };
        let data = ScalarFnData { scalar_fn };
        Ok(ArrayParts::new(vtable, dtype, len, data, slots))
    }

    #[inline]
    fn arg_dtypes(slots: &[Option<ArrayRef>]) -> SmallVec<[DType; 4]> {
        let mut arg_dtypes = SmallVec::with_capacity(slots.len());
        for child in slots.iter().flatten() {
            arg_dtypes.push(child.dtype().clone());
        }
        arg_dtypes
    }

    #[inline]
    fn infer_len(slots: &[Option<ArrayRef>]) -> VortexResult<usize> {
        let Some(child) = slots.first().and_then(Option::as_ref) else {
            vortex_bail!("ScalarFnArray length cannot be inferred without children");
        };
        Ok(child.len())
    }

    #[inline]
    fn validate_arity(scalar_fn: &ScalarFnRef, child_count: usize) -> VortexResult<()> {
        let arity = scalar_fn.signature().arity();
        vortex_ensure!(
            arity.matches(child_count),
            "ScalarFnArray requires {arity} children, got {child_count}"
        );
        Ok(())
    }

    #[inline]
    fn validate_children_len(slots: &[Option<ArrayRef>], len: usize) -> VortexResult<()> {
        for child in slots.iter().flatten() {
            vortex_ensure!(
                child.len() == len,
                "ScalarFnArray must have children equal to the array length"
            );
        }
        Ok(())
    }
}
