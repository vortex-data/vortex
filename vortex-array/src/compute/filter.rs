use std::cmp::Ordering;
use std::ops::BitAnd;
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
    fn filter(&self, array: &Array, mask: &FilterMask) -> VortexResult<ArrayData>;
}

impl<E: Encoding> FilterFn<ArrayData> for E
where
    E: FilterFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn filter(&self, array: &ArrayData, mask: &FilterMask) -> VortexResult<ArrayData> {
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
pub fn filter(array: &ArrayData, mask: &FilterMask) -> VortexResult<ArrayData> {
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

fn filter_impl(array: &ArrayData, mask: &FilterMask) -> VortexResult<ArrayData> {
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
            if self.true_count == 0 {
                return BooleanBuffer::new_unset(self.len);
            }

            if self.true_count == self.len {
                return BooleanBuffer::new_set(self.len);
            }

            if let Some(indices) = self.indices.get() {
                let mut buf = BooleanBufferBuilder::new(self.len);
                // TODO(ngates): for dense indices, we can do better by collecting into u64s.
                buf.append_n(self.len, false);
                indices.iter().for_each(|idx| buf.set_bit(*idx, true));
                return BooleanBuffer::from(buf);
            }

            if let Some(slices) = self.slices.get() {
                let mut buf = BooleanBufferBuilder::new(self.len);
                for (start, end) in slices.iter().copied() {
                    buf.append_n(start - buf.len(), false);
                    buf.append_n(end - start, true);
                }
                if let Some((_, end)) = slices.last() {
                    buf.append_n(self.len - end, false);
                }
                debug_assert_eq!(buf.len(), self.len);
                return BooleanBuffer::from(buf);
            }

            vortex_panic!("No mask representation found")
        })
    }

    /// Constructs an indices vector from one of the other representations.
    fn indices(&self) -> &[usize] {
        self.indices.get_or_init(|| {
            if self.true_count == 0 {
                return vec![];
            }

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
                let mut iter = indices.iter().copied();

                // Handle empty input
                let Some(first) = iter.next() else {
                    return slices;
                };

                let mut start = first;
                let mut prev = first;
                for curr in iter {
                    if curr != prev + 1 {
                        slices.push((start, prev + 1));
                        start = curr;
                    }
                    prev = curr;
                }

                // Don't forget the last range
                slices.push((start, prev + 1));

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
        assert!(vec.iter().all(|&idx| idx < len));
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
        assert!(vec.iter().all(|&(b, e)| b < e && e <= len));
        let true_count = vec.iter().map(|(b, e)| e - b).sum();
        Self(Arc::new(Inner {
            buffer: Default::default(),
            indices: Default::default(),
            slices: OnceLock::from(vec),
            len,
            true_count,
            selectivity: true_count as f64 / len as f64,
        }))
    }

    /// Create a new [`FilterMask`] from the intersection of two indices slices.
    pub fn from_intersection_indices(
        len: usize,
        lhs: impl Iterator<Item = usize>,
        rhs: impl Iterator<Item = usize>,
    ) -> Self {
        let mut intersection = Vec::with_capacity(len);
        let mut lhs = lhs.peekable();
        let mut rhs = rhs.peekable();
        while let (Some(&l), Some(&r)) = (lhs.peek(), rhs.peek()) {
            match l.cmp(&r) {
                Ordering::Less => {
                    lhs.next();
                }
                Ordering::Greater => {
                    rhs.next();
                }
                Ordering::Equal => {
                    intersection.push(l);
                    lhs.next();
                    rhs.next();
                }
            }
        }
        Self::from_indices(len, intersection)
    }

    #[inline]
    // There is good definition of is_empty, does it mean len == 0 or true_count == 0?
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.0.len
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
    pub fn iter(&self) -> FilterIter {
        if self.selectivity() > FILTER_SLICES_SELECTIVITY_THRESHOLD {
            FilterIter::Slices(self.slices())
        } else {
            FilterIter::Indices(self.indices())
        }
    }

    /// Slice the mask.
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        if self.true_count() == 0 {
            return Self::new_false(length);
        }
        if self.true_count() == self.len() {
            return Self::new_true(length);
        }

        if let Some(buffer) = self.0.buffer.get() {
            return Self::from_buffer(buffer.slice(offset, length));
        }

        let end = offset + length;

        if let Some(indices) = self.0.indices.get() {
            let indices = indices
                .iter()
                .copied()
                .filter(|&idx| offset <= idx && idx < end)
                .map(|idx| idx - offset)
                .collect();
            return Self::from_indices(length, indices);
        }

        if let Some(slices) = self.0.slices.get() {
            let slices = slices
                .iter()
                .copied()
                .filter(|(s, e)| *s < end && *e > offset)
                .map(|(s, e)| (s.max(offset), e.min(end)))
                .collect();
            return Self::from_slices(length, slices);
        }

        vortex_panic!("No mask representation found")
    }

    /// take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// We are more interested in low selectivity `self` (as indices) with a boolean buffer mask,
    /// so we don't optimize for other cases, yet.
    pub fn intersect_by_rank(&self, mask: &FilterMask) -> FilterMask {
        assert_eq!(self.true_count(), mask.len());

        if mask.true_count() == mask.len() {
            return self.clone();
        }

        if mask.true_count() == 0 {
            return Self::new_false(self.len());
        }

        // TODO(joe): support other fast paths, not converting self & mask into indices,
        // however indices are better for sparse masks, so this is the common case for now.
        let indices = self.0.indices();
        Self::from_indices(
            self.len(),
            mask.indices()
                .iter()
                .map(|idx|
                    // This is verified as safe because we know that the indices are less than the
                    // mask.len() and we known mask.len() <= self.len(),
                    // implied by `self.true_count() == mask.len()`.
                    unsafe{*indices.get_unchecked(*idx)})
                .collect(),
        )
    }
}

pub enum FilterIter<'a> {
    /// Slice of pre-cached indices of a filter mask.
    Indices(&'a [usize]),
    /// Slice of pre-cached slices of a filter mask.
    Slices(&'a [(usize, usize)]),
}

impl PartialEq for FilterMask {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.true_count() != other.true_count() {
            return false;
        }

        // Since the true counts are the same, a full or empty mask is equal to the other mask.
        if self.true_count() == 0 || self.true_count() == self.len() {
            return true;
        }

        // Compare the buffer if both masks are non-empty.
        if let (Some(buffer), Some(other)) = (self.0.buffer.get(), other.0.buffer.get()) {
            return buffer == other;
        }

        // Compare the indices if both masks are non-empty.
        if let (Some(indices), Some(other)) = (self.0.indices.get(), other.0.indices.get()) {
            return indices == other;
        }

        // Compare the slices if both masks are non-empty.
        if let (Some(slices), Some(other)) = (self.0.slices.get(), other.0.slices.get()) {
            return slices == other;
        }

        // Otherwise, we fall back to comparison based on sparsity.
        // We could go further an exhaustively check whose OnceLocks are initialized, but that's
        // probably not worth the effort.
        self.boolean_buffer() == other.boolean_buffer()
    }
}

impl Eq for FilterMask {}

impl BitAnd for &FilterMask {
    type Output = FilterMask;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("FilterMasks must have the same length");
        }
        if self.true_count() == 0 || rhs.true_count() == 0 {
            return FilterMask::new_false(self.len());
        }
        if self.true_count() == self.len() {
            return rhs.clone();
        }
        if rhs.true_count() == self.len() {
            return self.clone();
        }

        if let (Some(lhs), Some(rhs)) = (self.0.buffer.get(), rhs.0.buffer.get()) {
            return FilterMask::from_buffer(lhs & rhs);
        }

        if let (Some(lhs), Some(rhs)) = (self.0.indices.get(), rhs.0.indices.get()) {
            // TODO(ngates): this may only make sense for sparse indices.
            return FilterMask::from_intersection_indices(
                self.len(),
                lhs.iter().copied(),
                rhs.iter().copied(),
            );
        }

        // TODO(ngates): we could perform a more efficient intersection for slices.
        FilterMask::from_buffer(self.boolean_buffer() & rhs.boolean_buffer())
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
    use itertools::Itertools;

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

        let filtered = filter(&items, &mask).unwrap();
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

    #[test]
    fn filter_mask_all_true() {
        let mask = FilterMask::new_true(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.true_count(), 5);
        assert_eq!(mask.selectivity(), 1.0);
        assert_eq!(mask.indices(), &[0, 1, 2, 3, 4]);
        assert_eq!(mask.slices(), &[(0, 5)]);
        assert_eq!(mask.boolean_buffer(), &BooleanBuffer::new_set(5));
    }

    #[test]
    fn filter_mask_all_false() {
        let mask = FilterMask::new_false(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.true_count(), 0);
        assert_eq!(mask.selectivity(), 0.0);
        assert_eq!(mask.indices(), &[] as &[usize]);
        assert_eq!(mask.slices(), &[]);
        assert_eq!(mask.boolean_buffer(), &BooleanBuffer::new_unset(5));
    }

    #[test]
    fn filter_mask_from() {
        let masks = [
            FilterMask::from_indices(5, vec![0, 2, 3]),
            FilterMask::from_slices(5, vec![(0, 1), (2, 4)]),
            FilterMask::from_buffer(BooleanBuffer::from_iter([true, false, true, true, false])),
        ];

        for mask in &masks {
            assert_eq!(mask.len(), 5);
            assert_eq!(mask.true_count(), 3);
            assert_eq!(mask.selectivity(), 0.6);
            assert_eq!(mask.indices(), &[0, 2, 3]);
            assert_eq!(mask.slices(), &[(0, 1), (2, 4)]);
            assert_eq!(
                &mask.boolean_buffer().iter().collect_vec(),
                &[true, false, true, true, false]
            );
        }
    }

    #[test]
    fn filter_mask_eq() {
        assert_eq!(
            FilterMask::new_true(5),
            FilterMask::from_buffer(BooleanBuffer::new_set(5))
        );
        assert_eq!(
            FilterMask::new_false(5),
            FilterMask::from_buffer(BooleanBuffer::new_unset(5))
        );
        assert_eq!(
            FilterMask::from_indices(5, vec![0, 2, 3]),
            FilterMask::from_slices(5, vec![(0, 1), (2, 4)])
        );
        assert_eq!(
            FilterMask::from_indices(5, vec![0, 2, 3]),
            FilterMask::from_buffer(BooleanBuffer::from_iter([true, false, true, true, false]))
        );
    }

    #[test]
    fn filter_mask_intersect_all_as_bit_and() {
        let this =
            FilterMask::from_buffer(BooleanBuffer::from_iter(vec![true, true, true, true, true]));
        let mask = FilterMask::from_buffer(BooleanBuffer::from_iter(vec![
            false, true, false, true, true,
        ]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            FilterMask::from_indices(5, vec![1, 3, 4])
        );
    }

    #[test]
    fn filter_mask_intersect_all_true() {
        let this = FilterMask::from_buffer(BooleanBuffer::from_iter(vec![
            false, false, true, true, true,
        ]));
        let mask = FilterMask::from_buffer(BooleanBuffer::from_iter(vec![true, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            FilterMask::from_indices(5, vec![2, 3, 4])
        );
    }

    #[test]
    fn filter_mask_intersect_true() {
        let this = FilterMask::from_buffer(BooleanBuffer::from_iter(vec![
            true, false, false, true, true,
        ]));
        let mask = FilterMask::from_buffer(BooleanBuffer::from_iter(vec![true, false, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            FilterMask::from_indices(5, vec![0, 4])
        );
    }

    #[test]
    fn filter_mask_intersect_false() {
        let this = FilterMask::from_buffer(BooleanBuffer::from_iter(vec![
            true, false, false, true, true,
        ]));
        let mask = FilterMask::from_buffer(BooleanBuffer::from_iter(vec![false, false, false]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            FilterMask::from_indices(5, vec![])
        );
    }
}
