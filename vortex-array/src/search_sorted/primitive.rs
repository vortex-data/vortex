// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;
use std::cmp::Ordering;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::NativePType;
use crate::search_sorted::IndexOrd;

pub struct SearchSortedPrimitiveArray<'a, T>(
    &'a ArrayRef,
    RefCell<&'a mut ExecutionCtx>,
    PhantomData<T>,
);

impl<'a, T: NativePType> SearchSortedPrimitiveArray<'a, T> {
    pub fn new(array: &'a ArrayRef, ctx: &'a mut ExecutionCtx) -> Self {
        assert_eq!(
            array.dtype().as_ptype(),
            T::PTYPE,
            "Array PType must match primitive type"
        );
        Self(array, RefCell::new(ctx), PhantomData)
    }

    fn value(&self, idx: usize) -> VortexResult<T> {
        let ctx_mut = &mut self.1.borrow_mut();
        Ok(self
            .0
            .execute_scalar(idx, ctx_mut)?
            .as_primitive()
            .typed_value::<T>()
            .unwrap_or_else(|| T::zero()))
    }
}

impl<T: NativePType> IndexOrd<T> for SearchSortedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> VortexResult<Option<Ordering>> {
        let value = self.value(idx)?;
        Ok(Some(value.total_compare(*elem)))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

impl<T: NativePType> IndexOrd<Option<T>> for SearchSortedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &Option<T>) -> VortexResult<Option<Ordering>> {
        let ctx_mut = &mut self.1.borrow_mut();
        let value = self
            .0
            .is_valid(idx, ctx_mut)?
            .then(|| self.value(idx))
            .transpose()?;

        Ok(match (value, elem.as_ref()) {
            (Some(l), Some(r)) => Some(l.total_compare(*r)),
            (Some(_), None) => Some(Ordering::Greater),
            (None, Some(_)) => Some(Ordering::Less),
            (None, None) => Some(Ordering::Equal),
        })
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

impl<T: NativePType> IndexOrd<usize> for SearchSortedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &usize) -> VortexResult<Option<Ordering>> {
        let value = self.value(idx)?;

        let Some(elem_t) = T::from_usize(*elem) else {
            return Ok(Some(Ordering::Less));
        };

        Ok(Some(value.total_compare(elem_t)))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}
