// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::dtype::Nullability;
use crate::register_kernel;
use crate::scalar::Scalar;

impl MinMaxKernel for BoolVTable {
    fn min_max(&self, array: &BoolArray) -> VortexResult<Option<MinMaxResult>> {
        let mask = array.validity_mask()?;
        let true_non_null = match &mask {
            Mask::AllTrue(_) => array.to_bit_buffer(),
            Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => array.to_bit_buffer().bitand(v.bit_buffer()),
        };

        // TODO(ngates): we should be able to bail out earlier as soon as we have one true and
        //  one false value.
        let mut true_slices = true_non_null.set_slices();
        // If there are no slices, then all values are false
        // if there is a single slice that covers the entire array, then all values are true
        // otherwise, we have a mix of true and false values

        let Some(slice) = true_slices.next() else {
            // all false
            return Ok(Some(MinMaxResult {
                min: Scalar::bool(false, NonNullable),
                max: Scalar::bool(false, NonNullable),
            }));
        };
        if slice.0 == 0 && slice.1 == array.len() {
            // all true
            return Ok(Some(MinMaxResult {
                min: Scalar::bool(true, NonNullable),
                max: Scalar::bool(true, NonNullable),
            }));
        };

        // If the non null true slice doesn't cover the whole array we need to check for valid false values
        match &mask {
            // if the mask is all true or all false we don't need to look for false values
            Mask::AllTrue(_) | Mask::AllFalse(_) => {}
            Mask::Values(v) => {
                let false_non_null = (!array.to_bit_buffer()).bitand(v.bit_buffer());
                let mut false_slices = false_non_null.set_slices();

                let Some(_) = false_slices.next() else {
                    // In this case we don't have any false values which means we are all true and null
                    return Ok(Some(MinMaxResult {
                        min: Scalar::bool(true, NonNullable),
                        max: Scalar::bool(true, NonNullable),
                    }));
                };
            }
        }

        Ok(Some(MinMaxResult {
            min: Scalar::bool(false, NonNullable),
            max: Scalar::bool(true, NonNullable),
        }))
    }
}

register_kernel!(MinMaxKernelAdapter(BoolVTable).lift());

#[cfg(test)]
mod tests {
    use Nullability::NonNullable;

    use crate::arrays::BoolArray;
    use crate::compute::MinMaxResult;
    use crate::compute::min_max;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    #[test]
    fn test_min_max_nulls() {
        assert_eq!(
            min_max(BoolArray::from_iter(vec![Some(true), Some(true), None, None]).as_ref())
                .unwrap(),
            Some(MinMaxResult {
                min: Scalar::bool(true, NonNullable),
                max: Scalar::bool(true, NonNullable),
            })
        );

        assert_eq!(
            min_max(BoolArray::from_iter(vec![None, Some(true), Some(true)]).as_ref()).unwrap(),
            Some(MinMaxResult {
                min: Scalar::bool(true, NonNullable),
                max: Scalar::bool(true, NonNullable),
            })
        );

        assert_eq!(
            min_max(BoolArray::from_iter(vec![None, Some(true), Some(true), None]).as_ref())
                .unwrap(),
            Some(MinMaxResult {
                min: Scalar::bool(true, NonNullable),
                max: Scalar::bool(true, NonNullable),
            })
        );

        assert_eq!(
            min_max(BoolArray::from_iter(vec![Some(false), Some(false), None, None]).as_ref())
                .unwrap(),
            Some(MinMaxResult {
                min: Scalar::bool(false, NonNullable),
                max: Scalar::bool(false, NonNullable),
            })
        );
    }
}
