use std::sync::Arc;

use itertools::Itertools;
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::array::{ListArray, ListEncoding};
use crate::compute::{scalar_at, slice, ComputeVTable, ScalarAtFn, SliceFn};
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

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
        list_scalar_of(slice(array, index, index + 1)?)
    }
}

impl SliceFn<ListArray> for ListEncoding {
    fn slice(&self, array: &ListArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ListArray::try_new(
            array.elements().clone(),
            slice(array.offsets(), start, stop + 1)?,
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

fn list_scalar_of(array: ArrayData) -> VortexResult<Scalar> {
    let list = array.into_list()?;
    let elem = list.elements();
    let scalars: Vec<Scalar> = (0..elem.len()).map(|i| scalar_at(&elem, i)).try_collect()?;
    println!("{:?}", Arc::new(list.dtype().clone()));
    println!("{:?}", scalars);
    println!("{:?}", scalars[0].dtype());
    let (dt, null) = list.dtype().as_list().vortex_expect("List dtype");
    Ok(Scalar::list(
        Arc::new(list.dtype().as_list().clone()),
        scalars,
    ))
}
