use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{FillNullFn, Operator, compare, fill_null};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::{DictArray, DictEncoding};

impl FillNullFn<&DictArray> for DictEncoding {
    fn fill_null(&self, array: &DictArray, fill_value: Scalar) -> VortexResult<ArrayRef> {
        // If the fill value exists in the dictionary, we can simply rewrite the null codes to
        // point to the value.
        let found_fill_values = compare(
            array.values(),
            &ConstantArray::new(fill_value.clone(), array.values().len()),
            Operator::Eq,
        )?
        .to_bool()?;

        let Some(first_fill_value) = found_fill_values.boolean_buffer().set_indices().next() else {
            // No fill values found, so we must canonicalize and fill_null.
            // TODO(ngates): compute kernels should all return Option<ArrayRef> to support this
            //  fall back.
            return fill_null(&array.to_canonical()?.into_array(), fill_value);
        };

        // Now we rewrite the nullable codes to point at the fill value.
        let codes = fill_null(
            array.codes(),
            Scalar::new(
                array.codes().dtype().clone(),
                ScalarValue::from(first_fill_value),
            ),
        )?;
        // And fill nulls in the values
        let values = fill_null(array.values(), fill_value)?;

        Ok(DictArray::try_new(codes, values)?.into_array())
    }
}
