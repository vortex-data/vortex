use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::{ListArray, ListEncoding};
use crate::compute::{scalar_at, slice, ComputeVTable, ScalarAtFn, SliceFn};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl ComputeVTable for ListEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<ListArray> for ListEncoding {
    fn scalar_at(&self, array: &ListArray, index: usize) -> VortexResult<Scalar> {
        list_scalar_of(array.elements_at(index)?)
    }
}

impl SliceFn<ListArray> for ListEncoding {
    fn slice(&self, array: &ListArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ListArray::try_new(
            array.elements(),
            slice(array.offsets(), start, stop + 1)?,
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

fn list_scalar_of(array: ArrayData) -> VortexResult<Scalar> {
    let scalars: Vec<Scalar> = (0..array.len())
        .map(|i| scalar_at(&array, i))
        .try_collect()?;

    Ok(Scalar::list(Arc::new(array.dtype().clone()), scalars))
}
