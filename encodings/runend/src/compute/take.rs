use vortex_array::array::PrimitiveArray;
use vortex_array::compute::{take, TakeFn};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl TakeFn<RunEndArray> for RunEndEncoding {
    #[allow(deprecated)]
    fn take(&self, array: &RunEndArray, indices: &ArrayData) -> VortexResult<ArrayData> {
        let primitive_indices = indices.clone().into_primitive()?;
        let usize_indices = match_each_integer_ptype!(primitive_indices.ptype(), |$P| {
            primitive_indices
                .into_maybe_null_slice::<$P>()
                .into_iter()
                .map(|idx| {
                    let usize_idx = idx as usize;
                    if usize_idx >= array.len() {
                        vortex_error::vortex_bail!(OutOfBounds: usize_idx, 0, array.len());
                    }

                    Ok(usize_idx + array.offset())
                })
                .collect::<VortexResult<Vec<usize>>>()?
        });
        let physical_indices = array
            .find_physical_indices(&usize_indices)?
            .into_iter()
            .map(|idx| idx as u64)
            .collect::<Vec<_>>();
        let physical_indices_array = PrimitiveArray::from(physical_indices).into_array();
        take(array.values(), &physical_indices_array)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{scalar_at, slice, take};
    use vortex_array::{IntoArrayVariant, ToArrayData};

    use crate::RunEndArray;

    pub(crate) fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from(vec![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).to_array(),
        )
        .unwrap()
    }

    #[test]
    fn ree_take() {
        let taken = take(
            ree_array().as_ref(),
            PrimitiveArray::from(vec![9, 8, 1, 3]).as_ref(),
        )
        .unwrap();
        assert_eq!(
            taken.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &[5, 5, 1, 4]
        );
    }

    #[test]
    fn ree_take_end() {
        let taken = take(
            ree_array().as_ref(),
            PrimitiveArray::from(vec![11]).as_ref(),
        )
        .unwrap();
        assert_eq!(
            taken.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &[5]
        );
    }

    #[test]
    #[should_panic]
    fn ree_take_out_of_bounds() {
        take(
            ree_array().as_ref(),
            PrimitiveArray::from(vec![12]).as_ref(),
        )
        .unwrap();
    }

    #[test]
    fn sliced_take() {
        let sliced = slice(ree_array().as_ref(), 4, 9).unwrap();
        let taken = take(
            sliced.as_ref(),
            PrimitiveArray::from(vec![1, 3, 4]).as_ref(),
        )
        .unwrap();

        assert_eq!(taken.len(), 3);
        assert_eq!(scalar_at(taken.as_ref(), 0).unwrap(), 4.into());
        assert_eq!(scalar_at(taken.as_ref(), 1).unwrap(), 2.into());
        assert_eq!(scalar_at(taken.as_ref(), 2).unwrap(), 5.into());
    }
}
