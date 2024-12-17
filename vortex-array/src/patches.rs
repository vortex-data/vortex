use std::fmt::Debug;

use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{match_each_integer_ptype, DType, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::aliases::hash_map::HashMap;
use crate::array::PrimitiveArray;
use crate::compute::{
    scalar_at, search_sorted, search_sorted_usize, search_sorted_usize_many, slice,
    subtract_scalar, take, FilterMask, SearchResult, SearchSortedSide,
};
use crate::stats::{ArrayStatistics, Stat};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, ArrayLen as _, IntoArrayData, IntoArrayVariant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchesMetadata {
    len: usize,
    indices_ptype: PType,
}

impl PatchesMetadata {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn indices_dtype(&self) -> DType {
        assert!(
            self.indices_ptype.is_unsigned_int(),
            "Patch indices must be unsigned integers"
        );
        DType::Primitive(self.indices_ptype, NonNullable)
    }
}

/// A helper for working with patched arrays.
#[derive(Debug, Clone)]
pub struct Patches {
    array_len: usize,
    indices: ArrayData,
    values: ArrayData,
}

impl Patches {
    pub fn new(array_len: usize, indices: ArrayData, values: ArrayData) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Patch indices and values must have the same length"
        );
        assert!(
            indices.dtype().is_unsigned_int(),
            "Patch indices must be unsigned integers"
        );
        assert!(
            indices.len() <= array_len,
            "Patch indices must be shorter than the array length"
        );
        assert!(!indices.is_empty(), "Patch indices must not be empty");
        if let Some(max) = indices.statistics().get_as_cast::<u64>(Stat::Max) {
            assert!(
                max < array_len as u64,
                "Patch indices {} are longer than the array length {}",
                max,
                array_len
            );
        }
        Self {
            array_len,
            indices,
            values,
        }
    }

    pub fn into_parts(self) -> (usize, ArrayData, ArrayData) {
        (self.array_len, self.indices, self.values)
    }

    pub fn array_len(&self) -> usize {
        self.array_len
    }

    pub fn num_patches(&self) -> usize {
        self.indices.len()
    }

    pub fn dtype(&self) -> &DType {
        self.values.dtype()
    }

    pub fn indices(&self) -> &ArrayData {
        &self.indices
    }

    pub fn into_indices(self) -> ArrayData {
        self.indices
    }

    pub fn values(&self) -> &ArrayData {
        &self.values
    }

    pub fn into_values(self) -> ArrayData {
        self.values
    }

    pub fn indices_ptype(&self) -> PType {
        PType::try_from(self.indices.dtype()).vortex_expect("primitive indices")
    }

    pub fn to_metadata(&self, len: usize, dtype: &DType) -> VortexResult<PatchesMetadata> {
        if self.indices.len() > len {
            vortex_bail!(
                "Patch indices {} are longer than the array length {}",
                self.indices.len(),
                len
            );
        }
        if self.values.dtype() != dtype {
            vortex_bail!(
                "Patch values dtype {} does not match array dtype {}",
                self.values.dtype(),
                dtype
            );
        }
        Ok(PatchesMetadata {
            len: self.indices.len(),
            indices_ptype: PType::try_from(self.indices.dtype()).vortex_expect("primitive indices"),
        })
    }

    /// Get the patched value at a given index if it exists.
    pub fn get_patched(&self, index: usize) -> VortexResult<Option<Scalar>> {
        if let Some(patch_idx) = self.search_index(index)?.to_found() {
            scalar_at(self.values(), patch_idx).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Return the insertion point of [index] in the [Self::indices].
    fn search_index(&self, index: usize) -> VortexResult<SearchResult> {
        search_sorted_usize(&self.indices, index, SearchSortedSide::Left)
    }

    /// Return the search_sorted result for the given target re-mapped into the original indices.
    pub fn search_sorted<T: Into<Scalar>>(
        &self,
        target: T,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        search_sorted(self.values(), target.into(), side).and_then(|sr| {
            let sidx = sr.to_offsets_index(self.indices().len());
            let index = usize::try_from(&scalar_at(self.indices(), sidx)?)?;
            Ok(match sr {
                // If we reached the end of patched values when searching then the result is one after the last patch index
                SearchResult::Found(i) => SearchResult::Found(if i == self.indices().len() {
                    index + 1
                } else {
                    index
                }),
                // If the result is NotFound we should return index that's one after the nearest not found index for the corresponding value
                SearchResult::NotFound(i) => {
                    SearchResult::NotFound(if i == 0 { index } else { index + 1 })
                }
            })
        })
    }

    /// Returns the minimum patch index
    pub fn min_index(&self) -> VortexResult<usize> {
        usize::try_from(&scalar_at(self.indices(), 0)?)
    }

    /// Returns the maximum patch index
    pub fn max_index(&self) -> VortexResult<usize> {
        usize::try_from(&scalar_at(self.indices(), self.indices().len() - 1)?)
    }

    /// Filter the patches by a mask, resulting in new patches for the filtered array.
    pub fn filter(&self, mask: FilterMask) -> VortexResult<Option<Self>> {
        if mask.is_empty() {
            return Ok(None);
        }

        let buffer = mask.to_boolean_buffer()?;
        let mut coordinate_indices: Vec<u64> = Vec::new();
        let mut value_indices = Vec::new();
        let mut last_inserted_index: usize = 0;

        let flat_indices = self.indices().clone().into_primitive()?;
        match_each_integer_ptype!(flat_indices.ptype(), |$I| {
            for (value_idx, coordinate) in flat_indices.into_maybe_null_slice::<$I>().into_iter().enumerate() {
                if buffer.value(coordinate as usize) {
                    // We count the number of truthy values between this coordinate and the previous truthy one
                    let adjusted_coordinate = buffer.slice(last_inserted_index, (coordinate as usize) - last_inserted_index).count_set_bits() as u64;
                    coordinate_indices.push(adjusted_coordinate + coordinate_indices.last().copied().unwrap_or_default());
                    last_inserted_index = coordinate as usize;
                    value_indices.push(value_idx as u64);
                }
            }
        });

        if coordinate_indices.is_empty() {
            return Ok(None);
        }

        let indices = PrimitiveArray::from(coordinate_indices).into_array();
        let values = take(self.values(), PrimitiveArray::from(value_indices))?;

        Ok(Some(Self::new(mask.len(), indices, values)))
    }

    /// Slice the patches by a range of the patched array.
    pub fn slice(&self, start: usize, stop: usize) -> VortexResult<Option<Self>> {
        let patch_start = self.search_index(start)?.to_index();
        let patch_stop = self.search_index(stop)?.to_index();

        if patch_start == patch_stop {
            return Ok(None);
        }

        // Slice out the values
        let values = slice(self.values(), patch_start, patch_stop)?;

        // Subtract the start value from the indices
        let indices = slice(self.indices(), patch_start, patch_stop)?;
        let indices = subtract_scalar(&indices, &Scalar::from(start).cast(indices.dtype())?)?;

        Ok(Some(Self::new(stop - start, indices, values)))
    }

    // https://docs.google.com/spreadsheets/d/1D9vBZ1QJ6mwcIvV5wIL0hjGgVchcEnAyhvitqWu2ugU
    const PREFER_MAP_WHEN_PATCHES_OVER_INDICES_LESS_THAN: f64 = 5.0;

    fn is_map_faster_than_search(&self, take_indices: &PrimitiveArray) -> bool {
        (self.num_patches() as f64 / take_indices.len() as f64)
            < Self::PREFER_MAP_WHEN_PATCHES_OVER_INDICES_LESS_THAN
    }

    /// Take the indices from the patches.
    pub fn take(&self, take_indices: &ArrayData) -> VortexResult<Option<Self>> {
        if take_indices.is_empty() {
            return Ok(None);
        }
        let take_indices = take_indices.clone().into_primitive()?;
        if self.is_map_faster_than_search(&take_indices) {
            self.take_map(take_indices)
        } else {
            self.take_search(take_indices)
        }
    }

    pub fn take_search(&self, take_indices: PrimitiveArray) -> VortexResult<Option<Self>> {
        let new_length = take_indices.len();
        let take_indices = match_each_integer_ptype!(take_indices.ptype(), |$P| {
            take_indices
                .into_maybe_null_slice::<$P>()
                .into_iter()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()?
        });

        let (values_indices, new_indices): (Vec<u64>, Vec<u64>) =
            search_sorted_usize_many(self.indices(), &take_indices, SearchSortedSide::Left)?
                .iter()
                .enumerate()
                .filter_map(|(idx_in_take, search_result)| {
                    search_result
                        .to_found()
                        .map(|patch_idx| (patch_idx as u64, idx_in_take as u64))
                })
                .unzip();

        if new_indices.is_empty() {
            return Ok(None);
        }

        let new_indices = PrimitiveArray::from_vec(new_indices, Validity::NonNullable).into_array();

        let values_indices =
            PrimitiveArray::from_vec(values_indices, Validity::NonNullable).into_array();
        let new_values = take(self.values(), values_indices)?;

        Ok(Some(Self::new(new_length, new_indices, new_values)))
    }

    pub fn take_map(&self, take_indices: PrimitiveArray) -> VortexResult<Option<Self>> {
        let indices = self.indices.clone().into_primitive()?;
        match_each_integer_ptype!(self.indices_ptype(), |$INDICES| {
            let indices = indices
                .maybe_null_slice::<$INDICES>();
            match_each_integer_ptype!(take_indices.ptype(), |$TAKE_INDICES| {
                let take_indices = take_indices
                    .maybe_null_slice::<$TAKE_INDICES>();

                let new_length = take_indices.len();
                let sparse_index_to_value_index: HashMap<$INDICES, usize> = indices
                    .iter()
                    .enumerate()
                    .map(|(value_index, sparse_index)| (*sparse_index, value_index))
                    .collect();
                let min_index = self.min_index()?;
                let max_index = self.max_index()?;
                let (new_sparse_indices, value_indices): (Vec<u64>, Vec<u64>) =
                    take_indices
                    .iter()
                    .map(|x| usize::try_from(*x))
                    .process_results(|iter| {
                        iter
                           .enumerate()
                           .filter(|(_, ti)| *ti >= min_index && *ti <= max_index)
                           .filter_map(|(new_sparse_index, take_sparse_index)| {
                               sparse_index_to_value_index
                                   .get(&<$INDICES>::try_from(take_sparse_index).ok().vortex_expect(
                                       "take_sparse_index is between min and max index",
                                   ))
                                   .map(|value_index| (new_sparse_index as u64, *value_index as u64))
                           })
                           .unzip()
                    })?;

                if new_sparse_indices.is_empty() {
                    return Ok(None);
                }

                Ok(Some(Patches::new(
                    new_length,
                    ArrayData::from(new_sparse_indices),
                    take(self.values(), ArrayData::from(value_indices))?,
                )))
            })
        })
    }

    pub fn map_values<F>(self, f: F) -> VortexResult<Self>
    where
        F: FnOnce(ArrayData) -> VortexResult<ArrayData>,
    {
        let values = f(self.values)?;
        if self.indices.len() != values.len() {
            vortex_bail!(
                "map_values must preserve length: expected {} received {}",
                self.indices.len(),
                values.len()
            )
        }
        Ok(Self::new(self.array_len, self.indices, values))
    }

    pub fn map_values_opt<F>(self, f: F) -> VortexResult<Option<Self>>
    where
        F: FnOnce(ArrayData) -> Option<ArrayData>,
    {
        let Some(values) = f(self.values) else {
            return Ok(None);
        };
        if self.indices.len() == values.len() {
            vortex_bail!(
                "map_values must preserve length: expected {} received {}",
                self.indices.len(),
                values.len()
            )
        }
        Ok(Some(Self::new(self.array_len, self.indices, values)))
    }
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};

    use crate::array::PrimitiveArray;
    use crate::compute::{FilterMask, SearchResult, SearchSortedSide};
    use crate::patches::Patches;
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn test_filter() {
        let patches = Patches::new(
            100,
            PrimitiveArray::from(vec![10u32, 11, 20]).into_array(),
            PrimitiveArray::from(vec![100, 110, 200]).into_array(),
        );

        let filtered = patches
            .filter(FilterMask::from_indices(100, [10u32, 20, 30]))
            .unwrap()
            .unwrap();

        let indices = filtered.indices().clone().into_primitive().unwrap();
        let values = filtered.values().clone().into_primitive().unwrap();
        assert_eq!(indices.maybe_null_slice::<u64>(), &[0, 1]);
        assert_eq!(values.maybe_null_slice::<i32>(), &[100, 200]);
    }

    #[fixture]
    fn patches() -> Patches {
        Patches::new(
            20,
            PrimitiveArray::from(vec![2u64, 9, 15]).into_array(),
            PrimitiveArray::from_vec(vec![33_i32, 44, 55], Validity::AllValid).into_array(),
        )
    }

    #[rstest]
    fn search_larger_than(patches: Patches) {
        let res = patches.search_sorted(66, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_less_than(patches: Patches) {
        let res = patches.search_sorted(22, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(2));
    }

    #[rstest]
    fn search_found(patches: Patches) {
        let res = patches.search_sorted(44, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::Found(9));
    }

    #[rstest]
    fn search_not_found_right(patches: Patches) {
        let res = patches.search_sorted(56, SearchSortedSide::Right).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_sliced(patches: Patches) {
        let sliced = patches.slice(7, 20).unwrap().unwrap();
        assert_eq!(
            sliced.search_sorted(22, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
    }

    #[test]
    fn search_right() {
        let patches = Patches::new(
            2,
            PrimitiveArray::from(vec![0u64]).into_array(),
            PrimitiveArray::from_vec(vec![0u8], Validity::AllValid).into_array(),
        );

        assert_eq!(
            patches.search_sorted(0, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(1)
        );
        assert_eq!(
            patches.search_sorted(1, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(1)
        );
    }
}
