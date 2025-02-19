mod to_arrow;

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::{ListArray, ListEncoding};
use crate::compute::{
    scalar_at, slice, MaskFn, MinMaxFn, MinMaxResult, ScalarAtFn, SliceFn, ToArrowFn,
};
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

impl ComputeVTable for ListEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<Array>> {
        Some(self)
    }
}

impl ScalarAtFn<ListArray> for ListEncoding {
    fn scalar_at(&self, array: &ListArray, index: usize) -> VortexResult<Scalar> {
        let elem = array.elements_at(index)?;
        let scalars: Vec<Scalar> = (0..elem.len()).map(|i| scalar_at(&elem, i)).try_collect()?;

        Ok(Scalar::list(
            Arc::new(elem.dtype().clone()),
            scalars,
            array.dtype().nullability(),
        ))
    }
}

impl SliceFn<ListArray> for ListEncoding {
    fn slice(&self, array: &ListArray, start: usize, stop: usize) -> VortexResult<Array> {
        Ok(ListArray::try_new(
            array.elements(),
            slice(array.offsets(), start, stop + 1)?,
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

impl MaskFn<ListArray> for ListEncoding {
    fn mask(&self, array: &ListArray, mask: Mask) -> VortexResult<Array> {
        ListArray::try_new(
            array.elements(),
            array.offsets(),
            array.validity().mask(&mask)?,
        )
        .map(IntoArray::into_array)
    }
}

impl MinMaxFn<ListArray> for ListEncoding {
    fn min_max(&self, _array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement list min max
        Ok(None)
    }
}

#[cfg(test)]
mod test {
    use crate::array::{ListArray, PrimitiveArray};
    use crate::compute::test_harness::test_mask;
    use crate::validity::Validity;
    use crate::IntoArray as _;

    #[test]
    fn test_mask_list() {
        let elements = PrimitiveArray::from_iter(0..35);
        let offsets = PrimitiveArray::from_iter([0, 5, 11, 18, 26, 35]);
        let validity = Validity::AllValid;
        let array = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
            .unwrap()
            .into_array();

        test_mask(array);
    }
}
