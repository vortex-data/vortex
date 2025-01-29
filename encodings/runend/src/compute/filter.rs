use std::cmp::min;
use std::ops::AddAssign;

use arrow_buffer::BooleanBuffer;
use num_traits::AsPrimitive;
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::{filter, FilterFn};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, Canonical, IntoArrayData, IntoArrayVariant};
use vortex_buffer::buffer_mut;
use vortex_dtype::{match_each_unsigned_integer_ptype, NativePType};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap};
use vortex_mask::Mask;

use crate::compute::take::take_indices_unchecked;
use crate::{RunEndArray, RunEndEncoding};

const FILTER_TAKE_THRESHOLD: f64 = 0.1;

impl FilterFn<RunEndArray> for RunEndEncoding {
    fn filter(&self, array: &RunEndArray, mask: &Mask) -> VortexResult<ArrayData> {
        match mask {
            Mask::AllTrue(_) => Ok(array.clone().into_array()),
            Mask::AllFalse(_) => Ok(Canonical::empty(array.dtype()).into()),
            Mask::Values(mask_values) => {
                let runs_ratio = mask_values.true_count() as f64 / array.ends().len() as f64;

                if runs_ratio < FILTER_TAKE_THRESHOLD || mask_values.true_count() < 25 {
                    // This strategy is directly proportional to the number of indices.
                    take_indices_unchecked(array, mask_values.indices())
                } else {
                    // This strategy ends up being close to fixed cost based on the number of runs,
                    // rather than the number of indices.
                    let primitive_run_ends = array.ends().into_primitive()?;
                    let (run_ends, values_mask) = match_each_unsigned_integer_ptype!(primitive_run_ends.ptype(), |$P| {
                        filter_run_end_primitive(
                            primitive_run_ends.as_slice::<$P>(),
                            array.offset() as u64,
                            array.len() as u64,
                            mask_values.boolean_buffer(),
                        )?
                    });
                    let values = filter(&array.values(), &values_mask)?;

                    RunEndArray::try_new(run_ends.into_array(), values).map(|a| a.into_array())
                }
            }
        }
    }
}

// We expose this function to our benchmarks.
pub fn filter_run_end(array: &RunEndArray, mask: &Mask) -> VortexResult<ArrayData> {
    let primitive_run_ends = array.ends().into_primitive()?;
    let (run_ends, values_mask) = match_each_unsigned_integer_ptype!(primitive_run_ends.ptype(), |$P| {
        filter_run_end_primitive(
            primitive_run_ends.as_slice::<$P>(),
            array.offset() as u64,
            array.len() as u64,
            mask.values().vortex_expect("AllTrue and AllFalse handled by filter fn").boolean_buffer(),
        )?
    });
    let values = filter(&array.values(), &values_mask)?;

    RunEndArray::try_new(run_ends.into_array(), values).map(|a| a.into_array())
}

// Code adapted from apache arrow-rs https://github.com/apache/arrow-rs/blob/b1f5c250ebb6c1252b4e7c51d15b8e77f4c361fa/arrow-select/src/filter.rs#L425
fn filter_run_end_primitive<R: NativePType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BooleanBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    let mut new_run_ends = buffer_mut![R::zero(); run_ends.len()];

    let mut start = 0u64;
    let mut j = 0;
    let mut count = R::zero();

    let new_mask: Mask = BooleanBuffer::collect_bool(run_ends.len(), |i| {
        let mut keep = false;
        let end = min(run_ends[i].as_() - offset, length);

        // Safety: predicate must be the same length as the array the ends have been taken from
        for pred in
            (start..end).map(|i| unsafe { mask.value_unchecked(i.try_into().vortex_unwrap()) })
        {
            count += <R as From<bool>>::from(pred);
            keep |= pred
        }
        // this is to avoid branching
        new_run_ends[j] = count;
        j += keep as usize;

        start = end;
        keep
    })
    .into();

    new_run_ends.truncate(j);
    Ok((
        PrimitiveArray::new(new_run_ends, Validity::NonNullable),
        new_mask,
    ))
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::slice;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_mask::Mask;

    use super::filter_run_end;
    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
    }

    #[test]
    fn run_end_filter() {
        let arr = ree_array();
        let filtered = filter_run_end(
            &arr,
            &Mask::from_iter([
                true, true, false, false, false, false, false, false, false, false, true, true,
            ]),
        )
        .unwrap();
        let filtered_run_end = RunEndArray::maybe_from(filtered).unwrap();

        assert_eq!(
            filtered_run_end
                .ends()
                .into_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [2, 4]
        );
        assert_eq!(
            filtered_run_end
                .values()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [1, 5]
        );
    }

    #[test]
    fn filter_sliced_run_end() {
        let arr = slice(ree_array(), 2, 7).unwrap();
        let filtered = filter_run_end(
            &RunEndArray::maybe_from(arr).unwrap(),
            &Mask::from_iter([true, false, false, true, true]),
        )
        .unwrap();
        let filtered_run_end = RunEndArray::try_from(filtered).unwrap();

        assert_eq!(
            filtered_run_end
                .ends()
                .into_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [1, 2, 3]
        );
        assert_eq!(
            filtered_run_end
                .values()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [1, 4, 2]
        );
    }
}
