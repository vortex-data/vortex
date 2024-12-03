use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::{ListArray, ListEncoding};
use crate::compute::{scalar_at, slice, ComputeVTable, ListMeanFn, ScalarAtFn, SliceFn};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl ComputeVTable for ListEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn list_mean_fn(&self) -> Option<&dyn ListMeanFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<ListArray> for ListEncoding {
    fn scalar_at(&self, array: &ListArray, index: usize) -> VortexResult<Scalar> {
        let elem = array.elements_at(index)?;
        let scalars: Vec<Scalar> = (0..elem.len()).map(|i| scalar_at(&elem, i)).try_collect()?;

        Ok(Scalar::list(Arc::new(elem.dtype().clone()), scalars))
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

impl ListMeanFn<ListArray> for ListEncoding {
    fn list_mean(&self, _array: &ListArray) -> VortexResult<ArrayData> {
        todo!()
        // let offsets = array.offsets();
        // let ends = slice(&offsets, 1, 0)?;
        // let begins = slice(&offsets, 0, ends.len())?;
        // let _lengths = sub(&ends, &begins)?;
        //
        // let sum_array: ArrayData = todo!();
        //
        // let (float_ptype, nullability) = match sum_array.dtype() {
        //     DType::Primitive(ptype, nullability) => match ptype {
        //         PType::U8 => (PType::F16, nullability.clone()),
        //         PType::U16 => (PType::F32, nullability.clone()),
        //         PType::U32 => (PType::F64, nullability.clone()),
        //         PType::U64 => (PType::F64, nullability.clone()),
        //         PType::I8 => (PType::F16, nullability.clone()),
        //         PType::I16 => (PType::F32, nullability.clone()),
        //         PType::I32 => (PType::F64, nullability.clone()),
        //         PType::I64 => (PType::F64, nullability.clone()),
        //         PType::F16 => (PType::F16, nullability.clone()),
        //         PType::F32 => (PType::F32, nullability.clone()),
        //         PType::F64 => (PType::F64, nullability.clone()),
        //     },
        //     _ => {
        //         vortex_bail!("Expected a primitive dtype, found {:?}", sum_array.dtype());
        //     }
        // };
        // let sum_float_array = try_cast(&sum_array, &DType::Primitive(float_ptype, nullability))?;
        //
        // let mean_array = div(&sum_float_array, &lengths)?;
        // Ok(mean_array)
    }
}
