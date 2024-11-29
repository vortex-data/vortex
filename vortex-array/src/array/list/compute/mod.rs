use vortex_error::VortexResult;
use vortex_scalar::{ListScalar, Scalar};

use crate::array::{ListArray, ListEncoding};
use crate::compute::unary::{scalar_at, ScalarAtFn};
use crate::compute::{slice, ComputeVTable};
use crate::{ArrayDType, ArrayData, IntoArrayVariant};
use crate::encoding::ArrayEncodingRef;

impl ComputeVTable for ListEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<ListArray> for ListEncoding {
    fn scalar_at(&self, array: &ListArray, index: usize) -> VortexResult<Scalar> {
        list_scalar_of(slice(array, index, index + 1)?);
    }
}

fn list_scalar_of(array: ArrayData) -> VortexResult<ListScalar> {
    array.encoding().
    array.into_list()
    let elem = array.elements();
    let scalars = (0..elem.len()).map(|i| scalar_at(elem, i)).collect();
    Ok(Scalar::from(scalars))
}
