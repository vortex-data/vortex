use arrow_buffer::BooleanBuffer;
use vortex_dtype::Nullability;
use vortex_error::{vortex_err, VortexResult};

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::unary::FillForwardFn;
use crate::validity::{ArrayValidity, Validity};
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, ToArrayData};

impl FillForwardFn<BoolArray> for BoolEncoding {
    fn fill_forward(&self, array: &BoolArray) -> VortexResult<ArrayData> {
        let validity = array.logical_validity();
        // nothing to see or do in this case
        if array.dtype().nullability() == Nullability::NonNullable {
            return Ok(array.to_array());
        }

        // all valid, but we need to convert to non-nullable
        if validity.all_valid() {
            return Ok(BoolArray::new(array.boolean_buffer(), Nullability::Nullable).into_array());
        }
        // all invalid => fill with default value (false)
        if validity.all_invalid() {
            return Ok(BoolArray::try_new(
                BooleanBuffer::new_unset(array.len()),
                Validity::AllValid,
            )?
            .into_array());
        }

        let validity = validity
            .to_null_buffer()?
            .ok_or_else(|| vortex_err!("Failed to convert array validity to null buffer"))?;

        let bools = array.boolean_buffer();
        let mut last_value = false;
        let buffer = BooleanBuffer::from_iter(bools.iter().zip(validity.inner().iter()).map(
            |(v, valid)| {
                if valid {
                    last_value = v;
                }
                last_value
            },
        ));
        Ok(BoolArray::try_new(buffer, Validity::AllValid)?.into_array())
    }
}

#[cfg(test)]
mod test {
    use crate::array::BoolArray;
    use crate::validity::Validity;
    use crate::{compute, IntoArrayData};

    #[test]
    fn fill_forward() {
        let barr =
            BoolArray::from_iter(vec![None, Some(false), None, Some(true), None]).into_array();
        let filled_bool =
            BoolArray::try_from(compute::unary::fill_forward(&barr).unwrap()).unwrap();
        assert_eq!(
            filled_bool.boolean_buffer().iter().collect::<Vec<bool>>(),
            vec![false, false, false, true, true]
        );
        assert_eq!(filled_bool.validity(), Validity::AllValid);
    }
}
