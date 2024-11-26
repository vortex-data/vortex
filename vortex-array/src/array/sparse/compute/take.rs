use std::convert::identity;

use itertools::Itertools;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::aliases::hash_map::HashMap;
use crate::array::primitive::PrimitiveArray;
use crate::array::sparse::SparseArray;
use crate::array::SparseEncoding;
use crate::compute::{take, TakeFn, TakeOptions};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

impl TakeFn<SparseArray> for SparseEncoding {
    fn take(
        &self,
        array: &SparseArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        let flat_indices = indices.clone().into_primitive()?;
        // if we are taking a lot of values we should build a hashmap
        let (positions, physical_take_indices) = if indices.len() > 128 {
            take_map(array, &flat_indices)?
        } else {
            take_search_sorted(array, &flat_indices)?
        };

        let taken_values = take(array.values(), physical_take_indices, options)?;

        Ok(SparseArray::try_new(
            positions.into_array(),
            taken_values,
            indices.len(),
            array.fill_scalar(),
        )?
        .into_array())
    }
}

fn take_map(
    array: &SparseArray,
    indices: &PrimitiveArray,
) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    let indices_map: HashMap<u64, u64> = array
        .resolved_indices()
        .iter()
        .enumerate()
        .map(|(i, r)| (*r as u64, i as u64))
        .collect();
    let min_index = array.min_index().unwrap_or_default() as u64;
    let max_index = array.max_index().unwrap_or_default() as u64;
    let (positions, patch_indices): (Vec<u64>, Vec<u64>) = match_each_integer_ptype!(indices.ptype(), |$P| {
        indices.maybe_null_slice::<$P>()
            .iter()
            .map(|pi| *pi as u64)
            .enumerate()
            .filter(|(_, pi)| *pi >= min_index && *pi <= max_index) // short-circuit
            .filter_map(|(i, pi)| indices_map.get(&pi).map(|phy_idx| (i as u64, phy_idx)))
            .unzip()
    });
    Ok((
        PrimitiveArray::from(positions),
        PrimitiveArray::from(patch_indices),
    ))
}

fn take_search_sorted(
    array: &SparseArray,
    indices: &PrimitiveArray,
) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    let min_index = array.min_index().unwrap_or_default() as u64;
    let max_index = array.max_index().unwrap_or_default() as u64;
    let resolved = match_each_integer_ptype!(indices.ptype(), |$P| {
        indices
            .maybe_null_slice::<$P>()
            .iter()
            .enumerate()
            .filter(|(_, i)| **i as u64 >= min_index && **i as u64 <= max_index) // short-circuit
            .map(|(pos, i)| {
                array
                    .search_index(*i as usize)
                    .map(|r| r.to_found().map(|ii| (pos as u64, ii as u64)))
            })
            .filter_map_ok(identity)
            .collect::<VortexResult<Vec<_>>>()?
    });

    let (positions, patch_indices): (Vec<u64>, Vec<u64>) = resolved.into_iter().unzip();
    Ok((
        PrimitiveArray::from(positions),
        PrimitiveArray::from(patch_indices),
    ))
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::array::sparse::compute::take::take_map;
    use crate::array::sparse::SparseArray;
    use crate::compute::{take, TakeOptions};
    use crate::validity::Validity;
    use crate::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};

    fn sparse_array() -> ArrayData {
        SparseArray::try_new(
            PrimitiveArray::from(vec![0u64, 37, 47, 99]).into_array(),
            PrimitiveArray::from_vec(vec![1.23f64, 0.47, 9.99, 3.5], Validity::AllValid)
                .into_array(),
            100,
            Scalar::null_typed::<f64>(),
        )
        .unwrap()
        .into_array()
    }

    #[test]
    fn sparse_take() {
        let sparse = sparse_array();
        let taken = SparseArray::try_from(
            take(
                sparse,
                vec![0, 47, 47, 0, 99].into_array(),
                TakeOptions::default(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            taken
                .indices()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u64>(),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            taken
                .values()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<f64>(),
            [1.23f64, 9.99, 9.99, 1.23, 3.5]
        );
    }

    #[test]
    fn nonexistent_take() {
        let sparse = sparse_array();
        let taken = SparseArray::try_from(
            take(sparse, vec![69].into_array(), TakeOptions::default()).unwrap(),
        )
        .unwrap();
        assert!(taken
            .indices()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<u64>()
            .is_empty());
        assert!(taken
            .values()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<f64>()
            .is_empty());
    }

    #[test]
    fn ordered_take() {
        let sparse = sparse_array();
        let taken = SparseArray::try_from(
            take(&sparse, vec![69, 37].into_array(), TakeOptions::default()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            taken
                .indices()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u64>(),
            [1]
        );
        assert_eq!(
            taken
                .values()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<f64>(),
            [0.47f64]
        );
        assert_eq!(taken.len(), 2);
    }

    #[test]
    fn test_take_map() {
        let sparse = SparseArray::try_from(sparse_array()).unwrap();
        let indices = PrimitiveArray::from((0u64..100).collect_vec());
        let (positions, patch_indices) = take_map(&sparse, &indices).unwrap();
        assert_eq!(
            positions.maybe_null_slice::<u64>(),
            sparse
                .indices()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u64>()
        );
        assert_eq!(patch_indices.maybe_null_slice::<u64>(), [0u64, 1, 2, 3]);
    }
}
