use std::iter::TrustedLen;
use std::sync::OnceLock;

use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, MutableBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};

use crate::array::BoolArray;
use crate::arrow::FromArrowArray;
use crate::encoding::Encoding;
use crate::stats::ArrayStatistics;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoCanonical};

/// If the filter selects more than this fraction of rows, iterate over slices instead of indices.
///
/// Threshold of 0.8 chosen based on Arrow Rust, which is in turn based on:
///   <https://dl.acm.org/doi/abs/10.1145/3465998.3466009>
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

pub trait FilterFn<Array> {
    /// Filter an array by the provided predicate.
    fn filter(&self, array: &Array, mask: FilterMask) -> VortexResult<ArrayData>;
}

impl<E: Encoding> FilterFn<ArrayData> for E
where
    E: FilterFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn filter(&self, array: &ArrayData, mask: FilterMask) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        FilterFn::filter(encoding, array_ref, mask)
    }
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
pub fn filter(array: &ArrayData, mask: FilterMask) -> VortexResult<ArrayData> {
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

    if let Some(filter_fn) = array.encoding().filter_fn() {
        filter_fn.filter(array, mask)
    } else {
        // Fallback: implement using Arrow kernels.
        log::debug!(
            "No filter implementation found for {}",
            array.encoding().id(),
        );

        let array_ref = array.clone().into_canonical()?.into_arrow()?;
        let mask_array = BooleanArray::new(mask.to_boolean_buffer()?, None);
        let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

        Ok(ArrayData::from_arrow(filtered, array.dtype().is_nullable()))
    }
}

/// Represents the mask argument to a filter function.
/// Internally this will cache the canonical representation of the mask if it is ever used.
#[derive(Debug)]
pub struct FilterMask {
    array: ArrayData,
    true_count: usize,
    range_selectivity: f64,
    indices: OnceLock<Vec<usize>>,
    slices: OnceLock<Vec<(usize, usize)>>,
    buffer: OnceLock<BooleanBuffer>,
}

/// We implement Clone manually to trigger population of our cached indices or slices.
/// By making the filter API take FilterMask by value, whenever it gets used multiple times
/// in a recursive function, we will cache the slices internally.
impl Clone for FilterMask {
    fn clone(&self) -> Self {
        if self.range_selectivity > FILTER_SLICES_SELECTIVITY_THRESHOLD {
            let _: VortexResult<_> = self
                .slices
                .get_or_try_init(|| Ok(self.boolean_buffer()?.set_slices().collect()));
        } else {
            let _: VortexResult<_> = self.indices.get_or_try_init(|| {
                let mut indices = Vec::with_capacity(self.true_count());
                indices.extend(self.boolean_buffer()?.set_indices());
                Ok(indices)
            });
        }

        Self {
            array: self.array.clone(),
            true_count: self.true_count,
            range_selectivity: self.range_selectivity,
            indices: self.indices.clone(),
            slices: self.slices.clone(),
            buffer: self.buffer.clone(),
        }
    }
}

/// Wrapper around Arrow's BitIndexIterator that knows its total length.
pub struct BitIndexIterator<'a> {
    inner: arrow_buffer::bit_iterator::BitIndexIterator<'a>,
    index: usize,
    trusted_len: usize,
}

impl<'a> BitIndexIterator<'a> {
    pub fn new(
        inner: arrow_buffer::bit_iterator::BitIndexIterator<'a>,
        trusted_len: usize,
    ) -> Self {
        Self {
            inner,
            index: 0,
            trusted_len,
        }
    }
}

impl<'a> Iterator for BitIndexIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.trusted_len - self.index;
        (remaining, Some(remaining))
    }
}

/// Safety: BitIndexIterator is TrustedLen because it knows its total length.
unsafe impl<'a> TrustedLen for BitIndexIterator<'a> {}
impl<'a> ExactSizeIterator for BitIndexIterator<'a> {}

pub enum FilterIter<'a> {
    // Slice of pre-cached indices of a filter mask.
    Indices(&'a [usize]),
    // Iterator over set bits of the filter mask's boolean buffer.
    IndicesIter(BitIndexIterator<'a>),
    // Slice of pre-cached slices of a filter mask.
    Slices(&'a [(usize, usize)]),
    // Iterator over contiguous ranges of set bits of the filter mask's boolean buffer.
    SlicesIter(arrow_buffer::bit_iterator::BitSliceIterator<'a>),
}

impl FilterMask {
    /// Create a new FilterMask where the given indices are set.
    pub fn from_indices<I: IntoIterator<Item = usize>>(length: usize, indices: I) -> Self {
        let mut buffer = MutableBuffer::new_null(length);
        indices
            .into_iter()
            .for_each(|idx| arrow_buffer::bit_util::set_bit(&mut buffer, idx));
        Self::from(BooleanBufferBuilder::new_from_buffer(buffer, length).finish())
    }

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

    /// Return the selectivity of the full mask.
    pub fn selectivity(&self) -> f64 {
        self.true_count as f64 / self.len() as f64
    }

    /// Return the selectivity of the range of true values of the mask.
    pub fn range_selectivity(&self) -> f64 {
        self.range_selectivity
    }

    /// Get the canonical representation of the mask.
    pub fn to_boolean_buffer(&self) -> VortexResult<BooleanBuffer> {
        log::debug!(
            "FilterMask: len {} selectivity: {} true_count: {}",
            self.len(),
            self.range_selectivity(),
            self.true_count,
        );
        self.boolean_buffer().cloned()
    }

    fn boolean_buffer(&self) -> VortexResult<&BooleanBuffer> {
        self.buffer.get_or_try_init(|| {
            Ok(self
                .array
                .clone()
                .into_canonical()?
                .into_bool()?
                .boolean_buffer())
        })
    }

    /// Returns the best iterator based on a selectivity threshold.
    ///
    /// Currently, this threshold is fixed at 0.8 based on Arrow Rust.
    pub fn iter(&self) -> VortexResult<FilterIter> {
        Ok(
            if self.range_selectivity > FILTER_SLICES_SELECTIVITY_THRESHOLD {
                // Iterate over slices
                if let Some(slices) = self.slices.get() {
                    FilterIter::Slices(slices.as_slice())
                } else {
                    FilterIter::SlicesIter(self.boolean_buffer()?.set_slices())
                }
            } else {
                // Iterate over indices
                if let Some(indices) = self.indices.get() {
                    FilterIter::Indices(indices.as_slice())
                } else {
                    FilterIter::IndicesIter(BitIndexIterator::new(
                        self.boolean_buffer()?.set_indices(),
                        self.true_count,
                    ))
                }
            },
        )
    }

    #[deprecated(note = "Move to using iter() instead")]
    pub fn iter_slices(&self) -> VortexResult<impl Iterator<Item = (usize, usize)> + '_> {
        Ok(self.boolean_buffer()?.set_slices())
    }

    #[deprecated(note = "Move to using iter() instead")]
    pub fn iter_indices(&self) -> VortexResult<impl Iterator<Item = usize> + '_> {
        Ok(self.boolean_buffer()?.set_indices())
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

        let selectivity = true_count as f64 / array.len() as f64;

        Ok(Self {
            array,
            true_count,
            range_selectivity: selectivity,
            indices: OnceLock::new(),
            slices: OnceLock::new(),
            buffer: OnceLock::new(),
        })
    }
}

impl From<BooleanBuffer> for FilterMask {
    fn from(value: BooleanBuffer) -> Self {
        Self::try_from(BoolArray::from(value).into_array())
            .vortex_expect("Failed to convert BooleanBuffer to FilterMask")
    }
}

impl FromIterator<bool> for FilterMask {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::from(BooleanBuffer::from_iter(iter))
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

        let filtered = filter(&items, mask).unwrap();
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
