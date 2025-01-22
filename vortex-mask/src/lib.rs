mod bitand;
mod eq;
mod intersect_by_rank;

use std::cmp::Ordering;
use std::sync::{Arc, OnceLock};

use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use itertools::Itertools;
use vortex_error::vortex_panic;

/// If the mask selects more than this fraction of rows, iterate over slices instead of indices.
///
/// Threshold of 0.8 chosen based on Arrow Rust, which is in turn based on:
///   <https://dl.acm.org/doi/abs/10.1145/3465998.3466009>
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

/// Represents a set of sorted unique positive integers.
///
/// A [`Mask`] can be constructed from various representations, and converted to various
/// others. Internally, these are cached.
#[derive(Clone, Debug)]
pub struct Mask(Arc<Inner>);

#[derive(Debug)]
struct Inner {
    // The three possible representations of the mask.
    buffer: OnceLock<BooleanBuffer>,
    indices: OnceLock<Vec<usize>>,
    slices: OnceLock<Vec<(usize, usize)>>,

    // Pre-computed values.
    len: usize,
    true_count: usize,
    // i.e., the fraction of values that are true
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
                debug_assert_eq!(buf.len(), self.len);
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
                debug_assert!(indices.is_sorted());
                assert_eq!(indices.len(), self.true_count);
                return indices;
            }

            if let Some(slices) = self.slices.get() {
                let mut indices = Vec::with_capacity(self.true_count);
                indices.extend(slices.iter().flat_map(|(start, end)| *start..*end));
                debug_assert!(indices.is_sorted());
                assert_eq!(indices.len(), self.true_count);
                return indices;
            }

            vortex_panic!("No mask representation found")
        })
    }

    /// Constructs a slices vector from one of the other representations.
    #[allow(clippy::cast_possible_truncation)]
    fn slices(&self) -> &[(usize, usize)] {
        self.slices.get_or_init(|| {
            if self.true_count == self.len {
                return vec![(0, self.len)];
            }

            if let Some(buffer) = self.buffer.get() {
                return buffer.set_slices().collect();
            }

            if let Some(indices) = self.indices.get() {
                // Expected number of contiguous slices assuming a uniform distribution of true values.
                let expected_num_slices =
                    (self.selectivity * (self.len - self.true_count + 1) as f64).round() as usize;
                let mut slices = Vec::with_capacity(expected_num_slices);
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

impl Mask {
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

    /// Create a new [`Mask`] from a [`BooleanBuffer`].
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

    /// Create a new [`Mask`] from a [`Vec<usize>`].
    pub fn from_indices(len: usize, vec: Vec<usize>) -> Self {
        let true_count = vec.len();
        assert!(vec.is_sorted(), "Mask indices must be sorted");
        assert!(
            vec.last().is_none_or(|&idx| idx < len),
            "Mask indices must be in bounds (len={len})"
        );
        Self(Arc::new(Inner {
            buffer: Default::default(),
            indices: OnceLock::from(vec),
            slices: Default::default(),
            len,
            true_count,
            selectivity: true_count as f64 / len as f64,
        }))
    }

    /// Create a new [`Mask`] from a [`Vec<(usize, usize)>`] where each range
    /// represents a contiguous range of true values.
    pub fn from_slices(len: usize, vec: Vec<(usize, usize)>) -> Self {
        Self::check_slices(len, &vec);
        Self::from_slices_unchecked(len, vec)
    }

    fn from_slices_unchecked(len: usize, vec: Vec<(usize, usize)>) -> Self {
        #[cfg(debug_assertions)]
        Self::check_slices(len, &vec);

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

    #[inline]
    // There is no good definition of is_empty, does it mean len == 0 or true_count == 0?
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
    pub fn iter(&self) -> Iter {
        if self.selectivity() > FILTER_SLICES_SELECTIVITY_THRESHOLD {
            Iter::Slices(self.slices())
        } else {
            Iter::Indices(self.indices())
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
                .skip_while(|idx| *idx < offset)
                .take_while(|idx| *idx < end)
                .map(|idx| idx - offset)
                .collect();
            return Self::from_indices(length, indices);
        }

        if let Some(slices) = self.0.slices.get() {
            let slices = slices
                .iter()
                .copied()
                .skip_while(|(_, e)| *e <= offset)
                .take_while(|(s, _)| *s < end)
                .map(|(s, e)| (s.max(offset), e.min(end)))
                .collect();
            return Self::from_slices_unchecked(length, slices);
        }

        vortex_panic!("No mask representation found")
    }
}

pub enum Iter<'a> {
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
        assert_eq!(mask.selectivity(), 1.0);
        assert_eq!(mask.indices(), &[0, 1, 2, 3, 4]);
        assert_eq!(mask.slices(), &[(0, 5)]);
        assert_eq!(mask.boolean_buffer(), &BooleanBuffer::new_set(5));
    }

    #[test]
    fn mask_all_false() {
        let mask = Mask::new_false(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.true_count(), 0);
        assert_eq!(mask.selectivity(), 0.0);
        assert_eq!(mask.indices(), &[] as &[usize]);
        assert_eq!(mask.slices(), &[]);
        assert_eq!(mask.boolean_buffer(), &BooleanBuffer::new_unset(5));
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
            assert_eq!(mask.selectivity(), 0.6);
            assert_eq!(mask.indices(), &[0, 2, 3]);
            assert_eq!(mask.slices(), &[(0, 1), (2, 4)]);
            assert_eq!(
                &mask.boolean_buffer().iter().collect::<Vec<_>>(),
                &[true, false, true, true, false]
            );
        }
    }
}
