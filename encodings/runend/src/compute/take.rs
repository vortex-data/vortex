use num_traits::{AsPrimitive, NumCast};
use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::search_sorted::{SearchResult, SearchSorted, SearchSortedSide};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_buffer::Buffer;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexResult, vortex_bail};

use crate::{RunEndArray, RunEndEncoding};

impl TakeKernel for RunEndEncoding {
    fn take(&self, array: &RunEndArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let primitive_indices = indices.to_primitive()?;

        let checked_indices = match_each_integer_ptype!(primitive_indices.ptype(), |$P| {
            primitive_indices
                .as_slice::<$P>()
                .iter()
                .copied()
                .map(|idx| {
                    let usize_idx = idx as usize;
                    if usize_idx >= array.len() {
                        vortex_bail!(OutOfBounds: usize_idx, 0, array.len());
                    }
                    Ok(usize_idx)
                })
                .collect::<VortexResult<Vec<_>>>()?
        });

        take_indices_unchecked(array, &checked_indices)
    }
}

register_kernel!(TakeKernelAdapter(RunEndEncoding).lift());

/// Perform a take operation on a RunEndArray by binary searching for each of the indices.
pub fn take_indices_unchecked<T: AsPrimitive<usize>>(
    array: &RunEndArray,
    indices: &[T],
) -> VortexResult<ArrayRef> {
    let ends = array.ends().to_primitive()?;
    let ends_len = ends.len();

    let physical_indices = match_each_integer_ptype!(ends.ptype(), |$I| {
        let end_slices = ends.as_slice::<$I>();
        indices
            .iter()
            .map(|idx| idx.as_() + array.offset())
            .map(|idx| {
                match <$I as NumCast>::from(idx) {
                    Some(idx) => end_slices.search_sorted(&idx, SearchSortedSide::Right),
                    None => {
                        // The idx is too large for $I, therefore it's out of bounds.
                        SearchResult::NotFound(ends_len)
                    }
                }
            })
            .map(|result| result.to_ends_index(ends_len) as u64)
            .collect::<Buffer<u64>>()
            .into_array()
    });

    take(array.values(), &physical_indices)
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::take;
    use vortex_array::{Array, ToCanonical};

    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
    }

    #[test]
    fn ree_take() {
        let taken = take(&ree_array(), &PrimitiveArray::from_iter([9, 8, 1, 3])).unwrap();
        assert_eq!(
            taken.to_primitive().unwrap().as_slice::<i32>(),
            &[5, 5, 1, 4]
        );
    }

    #[test]
    fn ree_take_end() {
        let taken = take(&ree_array(), &PrimitiveArray::from_iter([11])).unwrap();
        assert_eq!(taken.to_primitive().unwrap().as_slice::<i32>(), &[5]);
    }

    #[test]
    #[should_panic]
    fn ree_take_out_of_bounds() {
        take(&ree_array(), &PrimitiveArray::from_iter([12])).unwrap();
    }

    #[test]
    fn sliced_take() {
        let sliced = ree_array().slice(4, 9).unwrap();
        let taken = take(sliced.as_ref(), &PrimitiveArray::from_iter([1, 3, 4])).unwrap();

        assert_eq!(taken.len(), 3);
        assert_eq!(taken.scalar_at(0).unwrap(), 4.into());
        assert_eq!(taken.scalar_at(1).unwrap(), 2.into());
        assert_eq!(taken.scalar_at(2).unwrap(), 5.into());
    }
}
