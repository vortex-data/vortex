use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::array::{ListArray, ListEncoding};
use crate::compute::{
    div, list_sum, scalar_at, slice, sub, try_cast, ComputeVTable, ListFn, ScalarAtFn, SliceFn,
};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl ComputeVTable for ListEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn list_fn(&self) -> Option<&dyn ListFn<ArrayData>> {
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

impl ListFn<ListArray> for ListEncoding {
    fn sum(&self, _array: &ListArray) -> VortexResult<ArrayData> {
        todo!()
        // let offsets = array.offsets().into_primitive()?;
        // let elements = array.elements();
        //
        // let ends = offsets.maybe_null_slice::<i32>();
        // let mut sums = PrimitiveArray::<Float64Type>::builder(array.len() - 1);
        //
        // let elements = elements.into_primitive()?;
        // let elements = elements.maybe_null_slice::<f64>();
        //
        // // TODO(marko): This is going to be slow...
        // let mut start = 0;
        // for &end in ends.iter().skip(1) {
        //     sums.append_value(elements[start as usize..end as usize].iter().sum());
        //     start = end;
        // }
        //
        // let sums_array = sums.finish();
        // Ok(ArrayData::from_arrow(&sums_array, false))
    }

    fn mean(&self, array: &ListArray) -> VortexResult<ArrayData> {
        let offsets = array.offsets();
        let ends = slice(&offsets, 1, offsets.len())?;
        let begins = slice(&offsets, 0, offsets.len() - 1)?;
        let lengths = sub(&ends, &begins)?;

        let sum_array: ArrayData = list_sum(array)?;

        // Cast the sum array to a float type - the mean is always a float.
        let (float_ptype, nullability) = match sum_array.dtype() {
            DType::Primitive(ptype, nullability) => match ptype {
                PType::U8 => (PType::F16, *nullability),
                PType::U16 => (PType::F32, *nullability),
                PType::U32 => (PType::F64, *nullability),
                PType::U64 => (PType::F64, *nullability),
                PType::I8 => (PType::F16, *nullability),
                PType::I16 => (PType::F32, *nullability),
                PType::I32 => (PType::F64, *nullability),
                PType::I64 => (PType::F64, *nullability),
                PType::F16 => (PType::F16, *nullability),
                PType::F32 => (PType::F32, *nullability),
                PType::F64 => (PType::F64, *nullability),
            },
            _ => {
                vortex_bail!("Expected a primitive dtype, found {:?}", sum_array.dtype());
            }
        };

        let sum_float_array = try_cast(&sum_array, &DType::Primitive(float_ptype, nullability))?;
        let lengths_float_array = try_cast(&lengths, &DType::Primitive(float_ptype, nullability))?;
        let mean_array = div(&sum_float_array, &lengths_float_array)?;

        Ok(mean_array)
    }
}
