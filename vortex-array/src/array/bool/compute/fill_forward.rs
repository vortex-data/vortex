use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::FillForwardFn;
use crate::validity::Validity;
use crate::{Array, IntoArray};

impl FillForwardFn<BoolArray> for BoolEncoding {
    fn fill_forward(&self, array: &BoolArray) -> VortexResult<Array> {
        let validity = array.logical_validity()?;

        // nothing to see or do in this case
        if array.dtype().nullability() == Nullability::NonNullable {
            return Ok(array.clone().into_array());
        }

        match validity.boolean_buffer() {
            AllOr::All => {
                // all valid, but we need to convert to non-nullable
                Ok(BoolArray::new(array.boolean_buffer(), Nullability::Nullable).into_array())
            }
            AllOr::None => {
                // all invalid => fill with default value (false)
                Ok(
                    BoolArray::try_new(BooleanBuffer::new_unset(array.len()), Validity::AllValid)?
                        .into_array(),
                )
            }
            AllOr::Some(validity) => {
                let bools = array.boolean_buffer();
                let mut last_value = false;
                let buffer = BooleanBuffer::from_iter(bools.iter().zip_eq(validity.iter()).map(
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
    }
}

#[cfg(test)]
mod test {
    use crate::array::BoolArray;
    use crate::validity::Validity;
    use crate::{compute, IntoArray};

    #[test]
    fn fill_forward() {
        let barr =
            BoolArray::from_iter(vec![None, Some(false), None, Some(true), None]).into_array();
        let filled_bool = BoolArray::try_from(compute::fill_forward(&barr).unwrap()).unwrap();
        assert_eq!(
            filled_bool.boolean_buffer().iter().collect::<Vec<bool>>(),
            vec![false, false, false, true, true]
        );
        assert_eq!(filled_bool.validity(), Validity::AllValid);
    }
}
