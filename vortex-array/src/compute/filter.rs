use std::sync::OnceLock;

use arrow_array::BooleanArray;
use arrow_buffer::BooleanBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult,
};

use crate::array::BoolArray;
use crate::arrow::FromArrowArray;
use crate::stats::ArrayStatistics;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoCanonical};

pub trait FilterFn {
    /// Filter an array by the provided predicate.
    fn filter(&self, mask: &FilterMask) -> VortexResult<ArrayData>;
}

/// Return a new array by applying a boolean predicate to select items from a base Array.
///
/// # Performance
///
/// This function attempts to amortize the cost of copying
///
/// # Panics
///
/// The `predicate` must receive an Array with type non-nullable bool, and will panic if this is
/// not the case.
pub fn filter(array: &ArrayData, mask: &FilterMask) -> VortexResult<ArrayData> {
    if mask.len() != array.len() {
        vortex_bail!(
            "mask.len() is {}, does not equal array.len() of {}",
            mask.len(),
            array.len()
        );
    }

    // Fast-path for empty mask.
    if mask.true_count() == 0 {
        return Ok(Canonical::empty(array.dtype())?.into());
    }

    // Fast-path for full mask
    if mask.true_count() == mask.len() {
        return Ok(array.clone());
    }

    array.with_dyn(|a| {
        if let Some(filter_fn) = a.filter() {
            filter_fn.filter(mask)
        } else {
            // Fallback: implement using Arrow kernels.
            let array_ref = array.clone().into_canonical()?.into_arrow()?;
            let mask_array = BooleanArray::new(mask.to_boolean_buffer()?, None);
            let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

            Ok(ArrayData::from_arrow(filtered, array.dtype().is_nullable()))
        }
    })
}

/// Represents the mask argument to a filter function.
/// Internally this will cache the canonical representation of the mask if it is ever used.
#[derive(Debug)]
pub struct FilterMask {
    array: ArrayData,
    true_count: usize,
    buffer: OnceLock<BooleanBuffer>,
}

impl FilterMask {
    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Get the true count of the mask.
    pub fn true_count(&self) -> usize {
        self.true_count
    }

    /// Get the false count of the mask.
    pub fn false_count(&self) -> usize {
        self.array.len() - self.true_count
    }

    /// Return the selectivity of the mask.
    pub fn selectivity(&self) -> f64 {
        self.true_count as f64 / self.array.len() as f64
    }

    /// Get the canonical representation of the mask.
    pub fn to_boolean_buffer(&self) -> VortexResult<BooleanBuffer> {
        self.buffer
            .get_or_try_init(|| {
                Ok(self
                    .array
                    .clone()
                    .into_canonical()?
                    .into_bool()?
                    .boolean_buffer())
            })
            .cloned()
    }

    /// Returns an iterator over the set bits in this mask.
    pub fn iter_indices(&self) -> VortexResult<impl Iterator<Item = usize> + '_> {
        let _ = self.to_boolean_buffer()?; // Compute the buffer
        match self.buffer.get() {
            None => vortex_bail!("Failed to compute boolean buffer"),
            Some(buffer) => Ok(buffer.set_indices()),
        }
    }

    /// Iterator of contiguous ranges of set bits.
    /// Returns (usize, usize) each representing an interval where the corresponding bits are set.
    pub fn iter_slices(&self) -> VortexResult<impl Iterator<Item = (usize, usize)> + '_> {
        let _ = self.to_boolean_buffer()?; // Compute the buffer
        match self.buffer.get() {
            None => vortex_bail!("Failed to compute boolean buffer"),
            Some(buffer) => Ok(buffer.set_slices()),
        }
    }
}

impl TryFrom<ArrayData> for FilterMask {
    type Error = VortexError;

    fn try_from(array: ArrayData) -> Result<Self, Self::Error> {
        if array.dtype() != &DType::Bool(Nullability::NonNullable) {
            vortex_bail!(
                "mask must be non-nullable bool, has dtype {}",
                array.dtype(),
            );
        }
        let true_count = array
            .statistics()
            .compute_true_count()
            .ok_or_else(|| vortex_err!("Failed to compute true count for boolean array"))?;
        Ok(Self {
            array,
            true_count,
            buffer: OnceLock::new(),
        })
    }
}

impl From<BooleanBuffer> for FilterMask {
    fn from(value: BooleanBuffer) -> Self {
        Self::from(BoolArray::from(value))
    }
}

impl From<BoolArray> for FilterMask {
    fn from(array: BoolArray) -> Self {
        if array.dtype() != &DType::Bool(Nullability::NonNullable) {
            vortex_panic!(
                "mask must be non-nullable bool, has dtype {}",
                array.dtype(),
            );
        }
        let true_count = array
            .statistics()
            .compute_true_count()
            .vortex_expect("Failed to compute true count for boolean array");
        Self {
            array: array.into_array(),
            true_count,
            buffer: OnceLock::new(),
        }
    }
}

impl FromIterator<bool> for FilterMask {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::from(BoolArray::from_iter(iter))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::array::{BoolArray, PrimitiveArray};
    use crate::compute::filter::filter;
    use crate::{IntoArrayData, IntoCanonical};

    #[test]
    fn test_filter() {
        let items =
            PrimitiveArray::from_nullable_vec(vec![Some(0i32), None, Some(1i32), None, Some(2i32)])
                .into_array();
        let mask = FilterMask::try_from(
            BoolArray::from_iter([true, false, true, false, true]).into_array(),
        )
        .unwrap();

        let filtered = filter(&items, &mask).unwrap();
        assert_eq!(
            filtered
                .into_canonical()
                .unwrap()
                .into_primitive()
                .unwrap()
                .into_maybe_null_slice::<i32>(),
            vec![0i32, 1i32, 2i32]
        );
    }
}
