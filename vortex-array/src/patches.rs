use std::fmt::Debug;
use std::hash::Hash;

use serde::{Deserialize, Serialize};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{
    match_each_integer_ptype, match_each_unsigned_integer_ptype, DType, NativeIndexPType, PType,
};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::aliases::hash_map::HashMap;
use crate::array::PrimitiveArray;
use crate::compute::{
    scalar_at, search_sorted, search_sorted_usize, search_sorted_usize_many, slice,
    subtract_scalar, take, try_cast, FilterMask, SearchResult, SearchSortedSide,
};
use crate::stats::{ArrayStatistics, Stat};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

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
        assert!(indices.dtype().is_int(), "Patch indices must be integers");
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

    pub fn values(&self) -> &ArrayData {
        &self.values
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
        if let Some(patch_idx) =
            search_sorted_usize(self.indices(), index, SearchSortedSide::Left)?.to_found()
        {
            scalar_at(self.values(), patch_idx).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Return the search_sorted result for the given target re-mapped into the original indices.
    pub fn search_sorted<T: Into<Scalar>>(
        &self,
        target: T,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        Ok(match search_sorted(self.values(), target.into(), side)? {
            SearchResult::Found(idx) => SearchResult::Found(if idx == self.indices().len() {
                self.array_len()
            } else {
                usize::try_from(&scalar_at(self.indices(), idx)?)?
            }),
            SearchResult::NotFound(idx) => SearchResult::NotFound(if idx == self.indices().len() {
                self.array_len()
            } else {
                usize::try_from(&scalar_at(self.indices(), idx)?)?
            }),
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
        let patch_start =
            search_sorted_usize(self.indices(), start, SearchSortedSide::Left)?.to_index();
        let patch_stop =
            search_sorted_usize(self.indices(), stop, SearchSortedSide::Left)?.to_index();

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

    fn is_map_faster_than_search(&self, take_indices: &ArrayData) -> bool {
        (self.num_patches() as f64 / take_indices.len() as f64)
            < Self::PREFER_MAP_WHEN_PATCHES_OVER_INDICES_LESS_THAN
    }

    /// Take the indices from the patches.
    pub fn take(&self, take_indices: &ArrayData) -> VortexResult<Option<Self>> {
        if self.is_map_faster_than_search(take_indices) {
            self.take_map(take_indices)
        } else {
            self.take_search(take_indices)
        }
    }

    pub fn take_search(&self, take_indices: &ArrayData) -> VortexResult<Option<Self>> {
        let new_length = take_indices.len();

        if take_indices.is_empty() {
            return Ok(None);
        }

        // TODO(ngates): plenty of optimisations to be made here
        let take_indices =
            try_cast(take_indices, &DType::Primitive(PType::U64, NonNullable))?.into_primitive()?;

        let (values_indices, new_indices): (Vec<u64>, Vec<u64>) = search_sorted_usize_many(
            self.indices(),
            &take_indices
                .into_maybe_null_slice::<u64>()
                .into_iter()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            SearchSortedSide::Left,
        )?
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

    pub fn take_map(&self, take_indices: &ArrayData) -> VortexResult<Option<Self>> {
        let take_indices = take_indices.clone().into_primitive()?;
        match_each_unsigned_integer_ptype!(self.indices_ptype(), |$INDICES| {
            let indices = self.indices
                .clone()
                .into_primitive()?;
            let indices = indices
                .maybe_null_slice::<$INDICES>();
            match_each_unsigned_integer_ptype!(take_indices.ptype(), |$TAKE_INDICES| {
                let take_indices = take_indices
                    .into_primitive()?;
                let take_indices = take_indices
                    .maybe_null_slice::<$TAKE_INDICES>();
                return self.take_map_helper(indices, take_indices);
            });
        });
    }

    fn take_map_helper<I: NativeIndexPType + Hash + Eq, TI: NativeIndexPType>(
        &self,
        indices: &[I],
        take_indices: &[TI],
    ) -> VortexResult<Option<Self>> {
        let new_length = take_indices.len();
        let sparse_index_to_value_index: HashMap<I, usize> = indices
            .iter()
            .enumerate()
            .map(|(value_index, sparse_index)| (*sparse_index, value_index))
            .collect();
        let min_index = self.min_index()?;
        let max_index = self.max_index()?;
        let (new_sparse_indices, value_indices): (Vec<u64>, Vec<u64>) = take_indices
            .iter()
            .enumerate()
            .filter(|(_, ti)| ti.as_usize() < min_index || ti.as_usize() > max_index)
            .filter_map(|(new_sparse_index, take_sparse_index)| {
                sparse_index_to_value_index
                    .get(&I::usize_as(take_sparse_index.as_usize()))
                    // FIXME(DK): should we choose a small index width or should the compressor do that?
                    .map(|value_index| (new_sparse_index as u64, *value_index as u64))
            })
            .unzip();

        if new_sparse_indices.len() == 0 {
            return Ok(None);
        }

        Ok(Some(Patches::new(
            new_length,
            ArrayData::from(new_sparse_indices),
            take(self.values(), ArrayData::from(value_indices))?,
        )))
    }
}

#[cfg(test)]
mod test {
    use crate::array::PrimitiveArray;
    use crate::compute::FilterMask;
    use crate::patches::Patches;
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
}
