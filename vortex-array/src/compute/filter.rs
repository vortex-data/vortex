use std::sync::{Arc, OnceLock};

use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use vortex_dtype::{DType, Nullability};
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult,
};

use crate::array::ConstantArray;
use crate::arrow::FromArrowArray;
use crate::compute::scalar_at;
use crate::encoding::Encoding;
use crate::stats::ArrayStatistics;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoArrayVariant, IntoCanonical};

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
        let (array_ref, encoding) = array.downcast_array_ref::<E>()?;
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

    let true_count = mask.true_count();

    // Fast-path for empty mask.
    if true_count == 0 {
        return Ok(Canonical::empty(array.dtype())?.into());
    }

    // Fast-path for full mask
    if true_count == mask.len() {
        return Ok(array.clone());
    }

    let filtered = filter_impl(array, mask)?;

    debug_assert_eq!(
        filtered.len(),
        true_count,
        "Filter length mismatch {}",
        array.encoding().id()
    );
    debug_assert_eq!(
        filtered.dtype(),
        array.dtype(),
        "Filter dtype mismatch {}",
        array.encoding().id()
    );

    Ok(filtered)
}

fn filter_impl(array: &ArrayData, mask: FilterMask) -> VortexResult<ArrayData> {
    if let Some(filter_fn) = array.encoding().filter_fn() {
        return filter_fn.filter(array, mask);
    }

    // We can use scalar_at if the mask has length 1.
    if mask.true_count() == 1 && array.encoding().scalar_at_fn().is_some() {
        let idx = mask.first().vortex_expect("true_count == 1");
        return Ok(ConstantArray::new(scalar_at(array, idx)?, 1).into_array());
    }

    // Fallback: implement using Arrow kernels.
    log::debug!(
        "No filter implementation found for {}",
        array.encoding().id(),
    );

    let array_ref = array.clone().into_arrow()?;
    let mask_array = BooleanArray::new(mask.boolean_buffer().clone(), None);
    let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

    Ok(ArrayData::from_arrow(filtered, array.dtype().is_nullable()))
}

/// Represents the mask argument to a filter function.
///
/// A [`FilterMask`] can be constructed from various representations, and converted to various
/// others. Internally, these are cached.
#[derive(Clone, Debug)]
pub struct FilterMask(Arc<Inner>);

#[derive(Debug)]
struct Inner {
    // The three possible representations of the mask.
    buffer: OnceLock<BooleanBuffer>,
    indices: OnceLock<Vec<usize>>,
    slices: OnceLock<Vec<(usize, usize)>>,

    // Pre-computed values.
    len: usize,
    true_count: usize,
    selectivity: f64,
}

impl Inner {
    /// Constructs a [`BooleanBuffer`] from one of the other representations.
    fn buffer(&self) -> &BooleanBuffer {
        self.buffer.get_or_init(|| {
            if self.true_count == self.len {
                return BooleanBuffer::new_set(self.len);
            }

            if let Some(indices) = self.indices.get() {
                let mut buf = BooleanBufferBuilder::new(self.len);
                if indices.len() < self.len / 2 {
                    buf.append_n(self.len, false);
                    indices.iter().for_each(|idx| buf.set_bit(*idx, true));
                } else {
                    buf.append_n(self.len, true);
                    indices.iter().for_each(|idx| buf.set_bit(*idx, false));
                }
                return BooleanBuffer::from(buf);
            }

            if let Some(slices) = self.slices.get() {
                let mut buf = BooleanBufferBuilder::new(self.len);
                slices.iter().for_each(|(start, end)| {
                    buf.append_n(*start - buf.len(), false);
                    buf.append_n(end - start, true);
                });
                return BooleanBuffer::from(buf);
            }

            vortex_panic!("No mask representation found")
        })
    }

    /// Constructs an indices vector from one of the other representations.
    fn indices(&self) -> &[usize] {
        self.indices.get_or_init(|| {
            if self.true_count == self.len {
                return (0..self.len).collect();
            }

            if let Some(buffer) = self.buffer.get() {
                let mut indices = Vec::with_capacity(self.true_count);
                indices.extend(buffer.set_indices());
                return indices;
            }

            if let Some(slices) = self.slices.get() {
                let mut indices = Vec::with_capacity(self.true_count);
                indices.extend(slices.iter().flat_map(|(start, end)| *start..*end));
                return indices;
            }

            vortex_panic!("No mask representation found")
        })
    }

    /// Constructs a slices vector from one of the other representations.
    fn slices(&self) -> &[(usize, usize)] {
        self.slices.get_or_init(|| {
            if self.true_count == self.len {
                return vec![(0, self.len)];
            }

            if let Some(buffer) = self.buffer.get() {
                return buffer.set_slices().collect();
            }

            if let Some(indices) = self.indices.get() {
                let mut slices = Vec::with_capacity(self.true_count); // Upper bound
                let mut start = 0;
                let mut end = 0;
                for idx in indices {
                    if *idx == end {
                        end += 1;
                    } else {
                        if end >= start {
                            // Only push if we have a valid range
                            slices.push((start, end + 1));
                        }
                        start = *idx;
                        end = idx + 1;
                    }
                }
                if end >= start {
                    slices.push((start, end + 1));
                }
                return slices;
            }

            vortex_panic!("No mask representation found")
        })
    }

    fn first(&self) -> Option<usize> {
        if self.true_count == 0 {
            return None;
        }
        if self.true_count == self.len {
            return Some(0);
        }
        if let Some(buffer) = self.buffer.get() {
            return buffer.set_indices().next();
        }
        if let Some(indices) = self.indices.get() {
            return indices.first().copied();
        }
        if let Some(slices) = self.slices.get() {
            return slices.first().map(|(start, _)| *start);
        }
        None
    }
}

impl FilterMask {
    /// Create a new FilterMask where all values are set.
    pub fn new_true(length: usize) -> Self {
        Self(Arc::new(Inner {
            buffer: Default::default(),
            indices: Default::default(),
            slices: Default::default(),
            len: length,
            true_count: length,
            selectivity: 1.0,
        }))
    }

    /// Create a new FilterMask where no values are set.
    pub fn new_false(length: usize) -> Self {
        Self(Arc::new(Inner {
            buffer: Default::default(),
            indices: Default::default(),
            slices: Default::default(),
            len: length,
            true_count: 0,
            selectivity: 0.0,
        }))
    }

    /// Create a new [`FilterMask`] from a [`BooleanBuffer`].
    pub fn from_buffer(buffer: BooleanBuffer) -> Self {
        let true_count = buffer.count_set_bits();
        let len = buffer.len();
        Self(Arc::new(Inner {
            buffer: OnceLock::from(buffer),
            indices: Default::default(),
            slices: Default::default(),
            len,
            true_count,
            selectivity: true_count as f64 / len as f64,
        }))
    }

    /// Create a new [`FilterMask`] from a [`Vec<usize>`].
    pub fn from_indices(len: usize, vec: Vec<usize>) -> Self {
        let true_count = vec.len();
        debug_assert!(vec.iter().all(|&idx| idx < len));
        Self(Arc::new(Inner {
            buffer: Default::default(),
            indices: OnceLock::from(vec),
            slices: Default::default(),
            len,
            true_count,
            selectivity: true_count as f64 / len as f64,
        }))
    }

    /// Create a new [`FilterMask`] from a [`Vec<(usize, usize)>`] where each range
    /// represents a contiguous range of true values.
    pub fn from_slices(len: usize, vec: Vec<(usize, usize)>) -> Self {
        let true_count = vec.len();
        debug_assert!(vec.iter().all(|&(b, e)| b < e && e < len));
        Self(Arc::new(Inner {
            buffer: Default::default(),
            indices: Default::default(),
            slices: OnceLock::from(vec),
            len,
            true_count,
            selectivity: true_count as f64 / len as f64,
        }))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// Get the true count of the mask.
    #[inline]
    pub fn true_count(&self) -> usize {
        self.0.true_count
    }

    /// Get the false count of the mask.
    #[inline]
    pub fn false_count(&self) -> usize {
        self.len() - self.true_count()
    }

    /// Return the selectivity of the full mask.
    #[inline]
    pub fn selectivity(&self) -> f64 {
        self.0.selectivity
    }

    /// Get the canonical representation of the mask.
    pub fn boolean_buffer(&self) -> &BooleanBuffer {
        self.0.buffer()
    }

    /// Get the indices of the true values in the mask.
    pub fn indices(&self) -> &[usize] {
        self.0.indices()
    }

    /// Get the slices of the true values in the mask.
    pub fn slices(&self) -> &[(usize, usize)] {
        self.0.slices()
    }

    /// Returns the first true index in the mask.
    pub fn first(&self) -> Option<usize> {
        self.0.first()
    }

    /// Returns the best iterator based on a selectivity threshold.
    ///
    /// Currently, this threshold is fixed at 0.8 based on Arrow Rust.
    pub fn iter(&self) -> VortexResult<FilterIter> {
        if self.selectivity() > FILTER_SLICES_SELECTIVITY_THRESHOLD {
            return Ok(FilterIter::Slices(self.slices()));
        }
        Ok(FilterIter::Indices(self.indices()))
    }
}

pub enum FilterIter<'a> {
    /// Slice of pre-cached indices of a filter mask.
    Indices(&'a [usize]),
    /// Slice of pre-cached slices of a filter mask.
    Slices(&'a [(usize, usize)]),
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

        if true_count == 0 {
            return Ok(Self::new_false(array.len()));
        }
        if true_count == array.len() {
            return Ok(Self::new_true(array.len()));
        }

        // TODO(ngates): should we have a `to_filter_mask` compute function where encodings
        //  pick the best possible conversion? E.g. SparseArray may want from_indices.
        Ok(Self::from_buffer(array.into_bool()?.boolean_buffer()))
    }
}

impl From<BooleanBuffer> for FilterMask {
    fn from(value: BooleanBuffer) -> Self {
        Self::from_buffer(value)
    }
}

impl FromIterator<bool> for FilterMask {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::from_buffer(BooleanBuffer::from_iter(iter))
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
            PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)])
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
                .as_slice::<i32>(),
            &[0i32, 1i32, 2i32]
        );
    }
}
