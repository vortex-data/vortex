use arrow_buffer::BooleanBuffer;
use vortex_dtype::Nullability;
use vortex_error::{vortex_err, VortexResult};

use crate::array::BoolArray;
use crate::compute::unary::FillForwardFn;
use crate::validity::{ArrayValidity, Validity};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl FillForwardFn for BoolArray {
    fn fill_forward(&self) -> VortexResult<ArrayData> {
        let validity = self.logical_validity();
        // nothing to see or do in this case
        if self.dtype().nullability() == Nullability::NonNullable {
            return Ok(self.clone().into());
        }
        // all valid, but we need to convert to non-nullable
        if validity.all_valid() {
            return Ok(
                Self::new(self.boolean_buffer().clone(), Nullability::Nullable).into_array(),
            );
        }
        // all invalid => fill with default value (false)
        if validity.all_invalid() {
            return Ok(
                Self::try_new(BooleanBuffer::new_unset(self.len()), Validity::AllValid)?
                    .into_array(),
            );
        }

        let validity = validity
            .to_null_buffer()?
            .ok_or_else(|| vortex_err!("Failed to convert array validity to null buffer"))?;

        let bools = self.boolean_buffer();
        let mut last_value = false;
        let buffer = BooleanBuffer::from_iter(bools.iter().zip(validity.inner().iter()).map(
            |(v, valid)| {
                if valid {
                    last_value = v;
                }
                last_value
            },
        ));
        Ok(Self::try_new(buffer, Validity::AllValid)?.into_array())
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
