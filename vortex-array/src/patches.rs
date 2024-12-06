use serde::{Deserialize, Serialize};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::array::PrimitiveArray;
use crate::compute::{
    filter, scalar_at, search_sorted, search_sorted_usize, search_sorted_usize_many, slice, take,
    try_cast, FilterMask, SearchResult, SearchSortedSide, TakeOptions,
};
use crate::stats::{ArrayStatistics, Stat};
use crate::validity::Validity;
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
            SearchResult::Found(idx) => {
                SearchResult::Found(usize::try_from(&scalar_at(self.indices(), idx)?)?)
            }
            SearchResult::NotFound(idx) => {
                SearchResult::NotFound(usize::try_from(&scalar_at(self.indices(), idx)?)?)
            }
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

    /// Filter the patches by a mask.
    /// FIXME(ngates): fix this cast
    #[allow(clippy::cast_possible_truncation)]
    pub fn filter(&self, mask: FilterMask) -> VortexResult<Option<Self>> {
        let mask = mask.to_boolean_buffer()?;

        let indices = self.indices().clone().into_primitive()?;

        let (patches_mask, new_indices): (Vec<usize>, Vec<u64>) = indices
            .maybe_null_slice::<u64>()
            .iter()
            .enumerate()
            .filter(|(_rank, idx)| mask.value(**idx as usize))
            .unzip();

        if new_indices.is_empty() {
            return Ok(None);
        }

        let new_indices = PrimitiveArray::from_vec(new_indices, Validity::NonNullable).into_array();
        let new_values = filter(
            self.values(),
            FilterMask::from_indices(self.array_len(), patches_mask),
        )?;

        Ok(Some(Self::new(mask.len(), new_indices, new_values)))
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

        Ok(Some(Self::new(
            stop - start,
            slice(self.indices(), patch_start, patch_stop)?,
            slice(self.values(), patch_start, patch_stop)?,
        )))
    }

    /// Take the indices from the patches.
    /// FIXME(ngates): fix this cast
    #[allow(clippy::cast_possible_truncation)]
    pub fn take(&self, indices: &ArrayData, _options: TakeOptions) -> VortexResult<Option<Self>> {
        if indices.is_empty() {
            return Ok(None);
        }

        // TODO(ngates): plenty of optimisations to be made here
        let take_indices =
            try_cast(indices, &DType::Primitive(PType::U64, NonNullable))?.into_primitive()?;
        let take_indices: Vec<usize> = take_indices
            .into_maybe_null_slice::<u64>()
            .into_iter()
            .map(|idx| idx as usize)
            .collect();

        let (values_indices, new_indices): (Vec<u64>, Vec<u64>) =
            search_sorted_usize_many(self.indices(), &take_indices, SearchSortedSide::Left)?
                .iter()
                .zip(take_indices)
                .filter_map(|(search_result, take_idx)| {
                    search_result
                        .to_found()
                        .map(|patch_idx| (patch_idx as u64, take_idx as u64))
                })
                .unzip();

        let new_indices = PrimitiveArray::from_vec(new_indices, Validity::NonNullable).into_array();

        let values_indices =
            PrimitiveArray::from_vec(values_indices, Validity::NonNullable).into_array();
        let new_values = take(self.values(), values_indices, TakeOptions::default())?;

        Ok(Some(Self::new(indices.len(), new_indices, new_values)))
    }
}
