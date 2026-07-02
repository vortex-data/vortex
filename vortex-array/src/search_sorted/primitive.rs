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

pub struct TypedPrimitiveArray<'a, T>(&'a ArrayRef, RefCell<&'a mut ExecutionCtx>, PhantomData<T>);

impl<'a, T: NativePType> TypedPrimitiveArray<'a, T> {
    pub fn new(array: &'a ArrayRef, ctx: &'a mut ExecutionCtx) -> Self {
        assert_eq!(
            array.dtype().as_ptype(),
            T::PTYPE,
            "Array PType must match primitive type"
        );
        Self(array, RefCell::new(ctx), PhantomData)
    }
}

impl<T: NativePType> IndexOrd<T> for TypedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> VortexResult<Option<Ordering>> {
        let value = self
            .0
            .execute_scalar(idx, &mut self.1.borrow_mut())?
            .as_primitive()
            .typed_value::<T>()
            .unwrap_or_else(|| T::zero());

        Ok(Some(value.total_compare(*elem)))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

impl<T: NativePType> IndexOrd<Option<T>> for TypedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &Option<T>) -> VortexResult<Option<Ordering>> {
        let ctx_mut = &mut self.1.borrow_mut();
        let value = self
            .0
            .is_valid(idx, ctx_mut)?
            .then(|| {
                self.0.execute_scalar(idx, ctx_mut).map(|s| {
                    s.as_primitive()
                        .typed_value::<T>()
                        .unwrap_or_else(|| T::zero())
                })
            })
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

impl<T: NativePType> IndexOrd<usize> for TypedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &usize) -> VortexResult<Option<Ordering>> {
        let value = self
            .0
            .execute_scalar(idx, &mut self.1.borrow_mut())?
            .as_primitive()
            .typed_value::<T>()
            .unwrap_or_else(|| T::zero());

        let Some(elem_t) = T::from_usize(*elem) else {
            return Ok(Some(Ordering::Less));
        };

        Ok(Some(value.total_compare(elem_t)))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}
