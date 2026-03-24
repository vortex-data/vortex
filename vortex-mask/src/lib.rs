// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A mask is a set of sorted unique positive integers.
#![deny(missing_docs)]

mod bitops;
mod eq;
mod intersect_by_rank;
mod iter_bools;
mod mask_mut;

#[cfg(feature = "arrow")]
mod arrow;
#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Bound;
use std::ops::RangeBounds;
use std::sync::Arc;
use std::sync::OnceLock;

use itertools::Itertools;
pub use mask_mut::*;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::set_bit_unchecked;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

/// Represents a set of values that are all included, all excluded, or some mixture of both.
pub enum AllOr<T> {
    /// All values are included.
    All,
    /// No values are included.
    None,
    /// Some values are included.
    Some(T),
}

impl<T> AllOr<T> {
    /// Returns the `Some` variant of the enum, or a default value.
    #[inline]
    pub fn unwrap_or_else<F, G>(self, all_true: F, all_false: G) -> T
    where
        F: FnOnce() -> T,
        G: FnOnce() -> T,
    {
        match self {
            Self::Some(v) => v,
            AllOr::All => all_true(),
            AllOr::None => all_false(),
        }
    }
}

impl<T> AllOr<&T> {
    /// Clone the inner value.
    #[inline]
    pub fn cloned(self) -> AllOr<T>
    where
        T: Clone,
    {
        match self {
            Self::All => AllOr::All,
            Self::None => AllOr::None,
            Self::Some(v) => AllOr::Some(v.clone()),
        }
    }
}

impl<T> Debug for AllOr<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => f.write_str("All"),
            Self::None => f.write_str("None"),
            Self::Some(v) => f.debug_tuple("Some").field(v).finish(),
        }
    }
}

impl<T> PartialEq for AllOr<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::All, Self::All) => true,
            (Self::None, Self::None) => true,
            (Self::Some(lhs), Self::Some(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl<T> Eq for AllOr<T> where T: Eq {}

/// Represents a set of sorted unique positive integers.
/// If a value is included in a Mask, it's valid.
///
/// A [`Mask`] can be constructed from various representations, and converted to various
/// others. Internally, these are cached.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub enum Mask {
    /// All values are included.
    AllTrue(usize),
    /// No values are included.
    AllFalse(usize),
    /// Some values are included, represented as a [`BitBuffer`].
    Values(Arc<MaskValues>),
}

impl Debug for Mask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllTrue(len) => write!(f, "All true({len})"),
            Self::AllFalse(len) => write!(f, "All false({len})"),
            Self::Values(mask) => write!(f, "{mask:?}"),
        }
    }
}

impl Default for Mask {
    fn default() -> Self {
        Self::new_true(0)
    }
}

/// Represents the values of a [`Mask`] that contains some true and some false elements.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaskValues {
    buffer: BitBuffer,

    // We cached the indices and slices representations, since it can be faster than iterating
    // the bit-mask over and over again.
    #[cfg_attr(feature = "serde", serde(skip))]
    indices: OnceLock<Vec<usize>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    slices: OnceLock<Vec<(usize, usize)>>,

    // Pre-computed values.
    true_count: usize,
    // i.e., the fraction of values that are true
    density: f64,
}

impl Debug for MaskValues {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "true_count={}, ", self.true_count)?;
        write!(f, "density={}, ", self.density)?;
        if let Some(v) = self.indices.get() {
            write!(f, "indices={v:?}, ")?;
        }
        if let Some(v) = self.slices.get() {
            write!(f, "slices={v:?}, ")?;
        }
        if f.alternate() {
            f.write_str("\n")?;
        }
        write!(f, "{}", self.buffer)
    }
}

impl Mask {
    /// Create a new Mask with the given length.
    pub fn new(length: usize, value: bool) -> Self {
        if value {
            Self::AllTrue(length)
        } else {
            Self::AllFalse(length)
        }
    }

    /// Create a new Mask where all values are set.
    #[inline]
    pub fn new_true(length: usize) -> Self {
        Self::AllTrue(length)
    }

    /// Create a new Mask where no values are set.
    #[inline]
    pub fn new_false(length: usize) -> Self {
        Self::AllFalse(length)
    }

    /// Create a new [`Mask`] from a [`BitBuffer`].
    pub fn from_buffer(buffer: BitBuffer) -> Self {
        let len = buffer.len();
        let true_count = buffer.true_count();

        if true_count == 0 {
            return Self::AllFalse(len);
        }
        if true_count == len {
            return Self::AllTrue(len);
        }

        Self::Values(Arc::new(MaskValues {
            buffer,
            indices: Default::default(),
            slices: Default::default(),
            true_count,
            density: true_count as f64 / len as f64,
        }))
    }

    /// Create a new [`Mask`] from a [`Vec<usize>`].
    // TODO(ngates): this should take an IntoIterator<usize>.
    pub fn from_indices(len: usize, indices: Vec<usize>) -> Self {
        let true_count = indices.len();
        assert!(indices.is_sorted(), "Mask indices must be sorted");
        assert!(
            indices.last().is_none_or(|&idx| idx < len),
            "Mask indices must be in bounds (len={len})"
        );

        if true_count == 0 {
            return Self::AllFalse(len);
        }
        if true_count == len {
            return Self::AllTrue(len);
        }

        let mut buf = BitBufferMut::new_unset(len);
        // TODO(ngates): for dense indices, we can do better by collecting into u64s.
        indices.iter().for_each(|&idx| buf.set(idx));
        debug_assert_eq!(buf.len(), len);

        Self::Values(Arc::new(MaskValues {
            buffer: buf.freeze(),
            indices: OnceLock::from(indices),
            slices: Default::default(),
            true_count,
            density: true_count as f64 / len as f64,
        }))
    }

    /// Create a new [`Mask`] from an [`IntoIterator<Item = usize>`] of indices to be excluded.
    pub fn from_excluded_indices(len: usize, indices: impl IntoIterator<Item = usize>) -> Self {
        let mut buf = BitBufferMut::new_set(len);

        let mut false_count: usize = 0;
        indices.into_iter().for_each(|idx| {
            buf.unset(idx);
            false_count += 1;
        });
        debug_assert_eq!(buf.len(), len);
        let true_count = len - false_count;

        // Return optimized variants when appropriate
        if false_count == 0 {
            return Self::AllTrue(len);
        }
        if false_count == len {
            return Self::AllFalse(len);
        }

        Self::Values(Arc::new(MaskValues {
            buffer: buf.freeze(),
            indices: Default::default(),
            slices: Default::default(),
            true_count,
            density: true_count as f64 / len as f64,
        }))
    }

    /// Create a new [`Mask`] from a [`Vec<(usize, usize)>`] where each range
    /// represents a contiguous range of true values.
    pub fn from_slices(len: usize, vec: Vec<(usize, usize)>) -> Self {
        Self::check_slices(len, &vec);
        Self::from_slices_unchecked(len, vec)
    }

    fn from_slices_unchecked(len: usize, slices: Vec<(usize, usize)>) -> Self {
        #[cfg(debug_assertions)]
        Self::check_slices(len, &slices);

        let true_count = slices.iter().map(|(b, e)| e - b).sum();
        if true_count == 0 {
            return Self::AllFalse(len);
        }
        if true_count == len {
            return Self::AllTrue(len);
        }

        let mut buf = BitBufferMut::new_unset(len);
        for (start, end) in slices.iter().copied() {
            (start..end).for_each(|idx| buf.set(idx));
        }
        debug_assert_eq!(buf.len(), len);

        Self::Values(Arc::new(MaskValues {
            buffer: buf.freeze(),
            indices: Default::default(),
            slices: OnceLock::from(slices),
            true_count,
            density: true_count as f64 / len as f64,
        }))
    }

    #[inline(always)]
    fn check_slices(len: usize, vec: &[(usize, usize)]) {
        assert!(vec.iter().all(|&(b, e)| b < e && e <= len));
        for (first, second) in vec.iter().tuple_windows() {
            assert!(
                first.0 < second.0,
                "Slices must be sorted, got {first:?} and {second:?}"
            );
            assert!(
                first.1 <= second.0,
                "Slices must be non-overlapping, got {first:?} and {second:?}"
            );
        }
    }

    /// Create a new [`Mask`] from the intersection of two indices slices.
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

    /// Clears the mask of all data. Drops any allocated capacity.
    pub fn clear(&mut self) {
        *self = Self::new_false(0);
    }

    /// Returns the length of the mask (not the number of true values).
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::AllTrue(len) => *len,
            Self::AllFalse(len) => *len,
            Self::Values(values) => values.len(),
        }
    }

    /// Returns true if the mask is empty i.e., it's length is 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::AllTrue(len) => *len == 0,
            Self::AllFalse(len) => *len == 0,
            Self::Values(values) => values.is_empty(),
        }
    }

    /// Get the true count of the mask.
    #[inline]
    pub fn true_count(&self) -> usize {
        match &self {
            Self::AllTrue(len) => *len,
            Self::AllFalse(_) => 0,
            Self::Values(values) => values.true_count,
        }
    }

    /// Get the false count of the mask.
    #[inline]
    pub fn false_count(&self) -> usize {
        match &self {
            Self::AllTrue(_) => 0,
            Self::AllFalse(len) => *len,
            Self::Values(values) => values.buffer.len() - values.true_count,
        }
    }

    /// Returns true if all values in the mask are true.
    #[inline]
    pub fn all_true(&self) -> bool {
        match &self {
            Self::AllTrue(_) => true,
            Self::AllFalse(0) => true,
            Self::AllFalse(_) => false,
            Self::Values(values) => values.buffer.len() == values.true_count,
        }
    }

    /// Returns true if all values in the mask are false.
    #[inline]
    pub fn all_false(&self) -> bool {
        self.true_count() == 0
    }

    /// Return the density of the full mask.
    #[inline]
    pub fn density(&self) -> f64 {
        match &self {
            Self::AllTrue(_) => 1.0,
            Self::AllFalse(_) => 0.0,
            Self::Values(values) => values.density,
        }
    }

    /// Returns the boolean value at a given index.
    ///
    /// ## Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn value(&self, idx: usize) -> bool {
        match self {
            Mask::AllTrue(_) => true,
            Mask::AllFalse(_) => false,
            Mask::Values(values) => values.buffer.value(idx),
        }
    }

    /// Returns the first true index in the mask.
    pub fn first(&self) -> Option<usize> {
        match &self {
            Self::AllTrue(len) => (*len > 0).then_some(0),
            Self::AllFalse(_) => None,
            Self::Values(values) => {
                if let Some(indices) = values.indices.get() {
                    return indices.first().copied();
                }
                if let Some(slices) = values.slices.get() {
                    return slices.first().map(|(start, _)| *start);
                }
                values.buffer.set_indices().next()
            }
        }
    }

    /// Returns the position in the mask of the nth true value.
    pub fn rank(&self, n: usize) -> usize {
        if n >= self.true_count() {
            vortex_panic!(
                "Rank {n} out of bounds for mask with true count {}",
                self.true_count()
            );
        }
        match &self {
            Self::AllTrue(_) => n,
            Self::AllFalse(_) => unreachable!("no true values in all-false mask"),
            // TODO(joe): optimize this function
            Self::Values(values) => values.indices()[n],
        }
    }

    /// Slice the mask.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len(),
        };

        assert!(start <= end);
        assert!(start <= self.len());
        assert!(end <= self.len());
        let len = end - start;

        match &self {
            Self::AllTrue(_) => Self::new_true(len),
            Self::AllFalse(_) => Self::new_false(len),
            Self::Values(values) => Self::from_buffer(values.buffer.slice(range)),
        }
    }

    /// Return the boolean buffer representation of the mask.
    #[inline]
    pub fn bit_buffer(&self) -> AllOr<&BitBuffer> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(&values.buffer),
        }
    }

    /// Return a boolean buffer representation of the mask, allocating new buffers for all-true
    /// and all-false variants.
    #[inline]
    pub fn to_bit_buffer(&self) -> BitBuffer {
        match self {
            Self::AllTrue(l) => BitBuffer::new_set(*l),
            Self::AllFalse(l) => BitBuffer::new_unset(*l),
            Self::Values(values) => values.bit_buffer().clone(),
        }
    }

    /// Return a boolean buffer representation of the mask, allocating new buffers for all-true
    /// and all-false variants.
    #[inline]
    pub fn into_bit_buffer(self) -> BitBuffer {
        match self {
            Self::AllTrue(l) => BitBuffer::new_set(l),
            Self::AllFalse(l) => BitBuffer::new_unset(l),
            Self::Values(values) => Arc::try_unwrap(values)
                .map(|v| v.into_bit_buffer())
                .unwrap_or_else(|v| v.bit_buffer().clone()),
        }
    }

    /// Return the indices representation of the mask.
    #[inline]
    pub fn indices(&self) -> AllOr<&[usize]> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(values.indices()),
        }
    }

    /// Return the slices representation of the mask.
    #[inline]
    pub fn slices(&self) -> AllOr<&[(usize, usize)]> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(values.slices()),
        }
    }

    /// Return an iterator over either indices or slices of the mask based on a density threshold.
    #[inline]
    pub fn threshold_iter(&self, threshold: f64) -> AllOr<MaskIter<'_>> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(values.threshold_iter(threshold)),
        }
    }

    /// Return [`MaskValues`] if the mask is not all true or all false.
    #[inline]
    pub fn values(&self) -> Option<&MaskValues> {
        if let Self::Values(values) = self {
            Some(values)
        } else {
            None
        }
    }

    /// Given monotonically increasing `indices` in [0, n_rows], returns the
    /// count of valid elements up to each index.
    ///
    /// This is O(n_rows).
    pub fn valid_counts_for_indices(&self, indices: &[usize]) -> Vec<usize> {
        match self {
            Self::AllTrue(_) => indices.to_vec(),
            Self::AllFalse(_) => vec![0; indices.len()],
            Self::Values(values) => {
                let mut bool_iter = values.bit_buffer().iter();
                let mut valid_counts = Vec::with_capacity(indices.len());
                let mut valid_count = 0;
                let mut idx = 0;
                for &next_idx in indices {
                    while idx < next_idx {
                        idx += 1;
                        valid_count += bool_iter
                            .next()
                            .unwrap_or_else(|| vortex_panic!("Row indices exceed array length"))
                            as usize;
                    }
                    valid_counts.push(valid_count);
                }

                valid_counts
            }
        }
    }

    /// Limit the mask to the first `limit` true values
    pub fn limit(self, limit: usize) -> Self {
        // Early return optimization: if we're asking for more true values than the total
        // length of the mask, then even if all values were true, we couldn't exceed the
        // limit, so return the original mask unchanged.
        if self.len() <= limit {
            return self;
        }

        match self {
            Mask::AllTrue(len) => {
                Self::from_iter([Self::new_true(limit), Self::new_false(len - limit)])
            }
            Mask::AllFalse(_) => self,
            Mask::Values(ref mask_values) => {
                if limit >= mask_values.true_count() {
                    return self;
                }

                let existing_buffer = mask_values.bit_buffer();

                let mut new_buffer_builder = BitBufferMut::new_unset(mask_values.len());
                debug_assert!(limit < mask_values.len());

                let ptr = new_buffer_builder.as_mut_ptr();
                for index in existing_buffer.set_indices().take(limit) {
                    // SAFETY: We checked that `limit` was less than the mask values length,
                    // therefore `index` must be within the bounds of the bit buffer.
                    unsafe { set_bit_unchecked(ptr, index) }
                }

                Self::from(new_buffer_builder.freeze())
            }
        }
    }

    /// Concatenate multiple masks together into a single mask.
    pub fn concat<'a>(masks: impl Iterator<Item = &'a Self>) -> VortexResult<Self> {
        let masks: Vec<_> = masks.collect();
        let len = masks.iter().map(|t| t.len()).sum();

        if masks.iter().all(|t| t.all_true()) {
            return Ok(Mask::AllTrue(len));
        }

        if masks.iter().all(|t| t.all_false()) {
            return Ok(Mask::AllFalse(len));
        }

        let mut builder = BitBufferMut::with_capacity(len);

        for mask in masks {
            match mask {
                Mask::AllTrue(n) => builder.append_n(true, *n),
                Mask::AllFalse(n) => builder.append_n(false, *n),
                Mask::Values(v) => builder.append_buffer(v.bit_buffer()),
            }
        }

        Ok(Mask::from_buffer(builder.freeze()))
    }
}

impl MaskValues {
    /// Returns the length of the mask.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the mask is empty i.e., it's length is 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the density of the mask.
    #[inline]
    pub fn density(&self) -> f64 {
        self.density
    }

    /// Returns the true count of the mask.
    #[inline]
    pub fn true_count(&self) -> usize {
        self.true_count
    }

    /// Returns the boolean buffer representation of the mask.
    #[inline]
    pub fn bit_buffer(&self) -> &BitBuffer {
        &self.buffer
    }

    /// Returns the boolean buffer representation of the mask.
    #[inline]
    pub fn into_bit_buffer(self) -> BitBuffer {
        self.buffer
    }

    /// Returns the boolean value at a given index.
    #[inline]
    pub fn value(&self, index: usize) -> bool {
        self.buffer.value(index)
    }

    /// Constructs an indices vector from one of the other representations.
    pub fn indices(&self) -> &[usize] {
        self.indices.get_or_init(|| {
            if self.true_count == 0 {
                return vec![];
            }

            if self.true_count == self.len() {
                return (0..self.len()).collect();
            }

            if let Some(slices) = self.slices.get() {
                let mut indices = Vec::with_capacity(self.true_count);
                indices.extend(slices.iter().flat_map(|(start, end)| *start..*end));
                debug_assert!(indices.is_sorted());
                assert_eq!(indices.len(), self.true_count);
                return indices;
            }

            let mut indices = Vec::with_capacity(self.true_count);
            indices.extend(self.buffer.set_indices());
            debug_assert!(indices.is_sorted());
            assert_eq!(indices.len(), self.true_count);
            indices
        })
    }

    /// Constructs a slices vector from one of the other representations.
    #[inline]
    pub fn slices(&self) -> &[(usize, usize)] {
        self.slices.get_or_init(|| {
            if self.true_count == self.len() {
                return vec![(0, self.len())];
            }

            self.buffer.set_slices().collect()
        })
    }

    /// Return an iterator over either indices or slices of the mask based on a density threshold.
    #[inline]
    pub fn threshold_iter(&self, threshold: f64) -> MaskIter<'_> {
        if self.density >= threshold {
            MaskIter::Slices(self.slices())
        } else {
            MaskIter::Indices(self.indices())
        }
    }

    /// Extracts the internal [`BitBuffer`].
    pub(crate) fn into_buffer(self) -> BitBuffer {
        self.buffer
    }
}

/// Iterator over the indices or slices of a mask.
pub enum MaskIter<'a> {
    /// Slice of pre-cached indices of a mask.
    Indices(&'a [usize]),
    /// Slice of pre-cached slices of a mask.
    Slices(&'a [(usize, usize)]),
}

impl From<BitBuffer> for Mask {
    fn from(value: BitBuffer) -> Self {
        Self::from_buffer(value)
    }
}

impl FromIterator<bool> for Mask {
    #[inline]
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::from_buffer(BitBuffer::from_iter(iter))
    }
}

impl FromIterator<Mask> for Mask {
    fn from_iter<T: IntoIterator<Item = Mask>>(iter: T) -> Self {
        let masks = iter
            .into_iter()
            .filter(|m| !m.is_empty())
            .collect::<Vec<_>>();
        let total_length = masks.iter().map(|v| v.len()).sum();

        // If they're all valid, then return a single validity.
        if masks.iter().all(|v| v.all_true()) {
            return Self::AllTrue(total_length);
        }
        // If they're all invalid, then return a single invalidity.
        if masks.iter().all(|v| v.all_false()) {
            return Self::AllFalse(total_length);
        }

        // Else, construct the boolean buffer
        let mut buffer = BitBufferMut::with_capacity(total_length);
        for mask in masks {
            match mask {
                Mask::AllTrue(count) => buffer.append_n(true, count),
                Mask::AllFalse(count) => buffer.append_n(false, count),
                Mask::Values(values) => {
                    buffer.append_buffer(values.bit_buffer());
                }
            };
        }
        Self::from_buffer(buffer.freeze())
    }
}
