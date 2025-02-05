#![feature(trusted_len)]
//! A mask is a set of sorted unique positive integers.
#![deny(missing_docs)]
mod bitand;
mod eq;
mod intersect_by_rank;
mod iter_bools;

use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, OnceLock};

use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use itertools::Itertools;

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
///
/// A [`Mask`] can be constructed from various representations, and converted to various
/// others. Internally, these are cached.
#[derive(Clone, Debug)]
pub enum Mask {
    /// All values are included.
    AllTrue(usize),
    /// No values are included.
    AllFalse(usize),
    /// Some values are included, represented as a [`BooleanBuffer`].
    Values(Arc<MaskValues>),
}

/// Represents the values of a [`Mask`] that contains some true and some false elements.
#[derive(Debug)]
pub struct MaskValues {
    buffer: BooleanBuffer,

    // We cached the indices and slices representations, since it can be faster than iterating
    // the bit-mask over and over again.
    indices: OnceLock<Vec<usize>>,
    slices: OnceLock<Vec<(usize, usize)>>,

    // Pre-computed values.
    true_count: usize,
    // i.e., the fraction of values that are true
    density: f64,
}

impl MaskValues {
    /// Returns the length of the mask.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the true count of the mask.
    pub fn true_count(&self) -> usize {
        self.true_count
    }

    /// Returns the boolean buffer representation of the mask.
    pub fn boolean_buffer(&self) -> &BooleanBuffer {
        &self.buffer
    }

    /// Returns the boolean value at a given index.
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
    #[allow(clippy::cast_possible_truncation)]
    pub fn slices(&self) -> &[(usize, usize)] {
        self.slices.get_or_init(|| {
            if self.true_count == self.len() {
                return vec![(0, self.len())];
            }

            self.buffer.set_slices().collect()
        })
    }

    /// Return an iterator over either indices or slices of the mask based on a density threshold.
    pub fn threshold_iter(&self, threshold: f64) -> MaskIter {
        if self.density >= threshold {
            MaskIter::Slices(self.slices())
        } else {
            MaskIter::Indices(self.indices())
        }
    }
}

impl Mask {
    /// Create a new Mask where all values are set.
    pub fn new_true(length: usize) -> Self {
        Self::AllTrue(length)
    }

    /// Create a new Mask where no values are set.
    pub fn new_false(length: usize) -> Self {
        Self::AllFalse(length)
    }

    /// Create a new [`Mask`] from a [`BooleanBuffer`].
    pub fn from_buffer(buffer: BooleanBuffer) -> Self {
        let len = buffer.len();
        let true_count = buffer.count_set_bits();

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

        let mut buf = BooleanBufferBuilder::new(len);
        // TODO(ngates): for dense indices, we can do better by collecting into u64s.
        buf.append_n(len, false);
        indices.iter().for_each(|idx| buf.set_bit(*idx, true));
        debug_assert_eq!(buf.len(), len);

        Self::Values(Arc::new(MaskValues {
            buffer: buf.finish(),
            indices: OnceLock::from(indices),
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

        let mut buf = BooleanBufferBuilder::new(len);
        for (start, end) in slices.iter().copied() {
            buf.append_n(start - buf.len(), false);
            buf.append_n(end - start, true);
        }
        if let Some((_, end)) = slices.last() {
            buf.append_n(len - end, false);
        }
        debug_assert_eq!(buf.len(), len);

        Self::Values(Arc::new(MaskValues {
            buffer: buf.finish(),
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
                "Slices must be sorted, got {:?} and {:?}",
                first,
                second
            );
            assert!(
                first.1 <= second.0,
                "Slices must be non-overlapping, got {:?} and {:?}",
                first,
                second
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

    /// Returns the length of the mask (not the number of true values).
    #[inline]
    // It's confusing to provide is_empty, does it mean len == 0 or true_count == 0?
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match &self {
            Self::AllTrue(len) => *len,
            Self::AllFalse(len) => *len,
            Self::Values(values) => values.buffer.len(),
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

    /// Slice the mask.
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        assert!(offset + length <= self.len());
        match &self {
            Self::AllTrue(_) => Self::new_true(length),
            Self::AllFalse(_) => Self::new_false(length),
            Self::Values(values) => Self::from_buffer(values.buffer.slice(offset, length)),
        }
    }

    /// Return the boolean buffer representation of the mask.
    pub fn boolean_buffer(&self) -> AllOr<&BooleanBuffer> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(&values.buffer),
        }
    }

    /// Return a boolean buffer representation of the mask, allocating new buffers for all-true
    /// and all-false variants.
    pub fn to_boolean_buffer(&self) -> BooleanBuffer {
        match self {
            Self::AllTrue(l) => BooleanBuffer::new_set(*l),
            Self::AllFalse(l) => BooleanBuffer::new_unset(*l),
            Self::Values(values) => values.boolean_buffer().clone(),
        }
    }

    /// Returns an Arrow null buffer representation of the mask.
    pub fn to_null_buffer(&self) -> Option<NullBuffer> {
        match self {
            Mask::AllTrue(_) => None,
            Mask::AllFalse(l) => Some(NullBuffer::new_null(*l)),
            Mask::Values(values) => Some(NullBuffer::from(values.buffer.clone())),
        }
    }

    /// Return the indices representation of the mask.
    pub fn indices(&self) -> AllOr<&[usize]> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(values.indices()),
        }
    }

    /// Return the slices representation of the mask.
    pub fn slices(&self) -> AllOr<&[(usize, usize)]> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(values.slices()),
        }
    }

    /// Return an iterator over either indices or slices of the mask based on a density threshold.
    pub fn threshold_iter(&self, threshold: f64) -> AllOr<MaskIter> {
        match &self {
            Self::AllTrue(_) => AllOr::All,
            Self::AllFalse(_) => AllOr::None,
            Self::Values(values) => AllOr::Some(values.threshold_iter(threshold)),
        }
    }

    /// Return [`MaskValues`] if the mask is not all true or all false.
    pub fn values(&self) -> Option<&MaskValues> {
        match self {
            Self::Values(values) => Some(values),
            _ => None,
        }
    }
}

/// Iterator over the indices or slices of a mask.
pub enum MaskIter<'a> {
    /// Slice of pre-cached indices of a mask.
    Indices(&'a [usize]),
    /// Slice of pre-cached slices of a mask.
    Slices(&'a [(usize, usize)]),
}

impl From<BooleanBuffer> for Mask {
    fn from(value: BooleanBuffer) -> Self {
        Self::from_buffer(value)
    }
}

impl FromIterator<bool> for Mask {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::from_buffer(BooleanBuffer::from_iter(iter))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn mask_all_true() {
        let mask = Mask::new_true(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.true_count(), 5);
        assert_eq!(mask.density(), 1.0);
        assert_eq!(mask.indices(), AllOr::All);
        assert_eq!(mask.slices(), AllOr::All);
        assert_eq!(mask.boolean_buffer(), AllOr::All,);
    }

    #[test]
    fn mask_all_false() {
        let mask = Mask::new_false(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.true_count(), 0);
        assert_eq!(mask.density(), 0.0);
        assert_eq!(mask.indices(), AllOr::None);
        assert_eq!(mask.slices(), AllOr::None);
        assert_eq!(mask.boolean_buffer(), AllOr::None,);
    }

    #[test]
    fn mask_from() {
        let masks = [
            Mask::from_indices(5, vec![0, 2, 3]),
            Mask::from_slices(5, vec![(0, 1), (2, 4)]),
            Mask::from_buffer(BooleanBuffer::from_iter([true, false, true, true, false])),
        ];

        for mask in &masks {
            assert_eq!(mask.len(), 5);
            assert_eq!(mask.true_count(), 3);
            assert_eq!(mask.density(), 0.6);
            assert_eq!(mask.indices(), AllOr::Some(&[0, 2, 3][..]));
            assert_eq!(mask.slices(), AllOr::Some(&[(0, 1), (2, 4)][..]));
            assert_eq!(
                mask.boolean_buffer(),
                AllOr::Some(&BooleanBuffer::from_iter([true, false, true, true, false]))
            );
        }
    }
}
