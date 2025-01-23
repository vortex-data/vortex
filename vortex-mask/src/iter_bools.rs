use std::iter;
use std::iter::{Peekable, TrustedLen};

use crate::Mask;

impl Mask {
    /// Provides a closure with an iterator over the boolean values of the mask.
    ///
    /// This allows us to provide different implementations of the iterator based on the underlying
    /// representation of the mask, while avoiding a heap allocation to return a boxed iterator.
    ///
    /// Note that bool iteration might not be the fastest way to achieve whatever is it you're
    /// trying to do!
    pub fn iter_bools<F, T>(&self, mut f: F) -> T
    where
        F: FnMut(&mut dyn Iterator<Item = bool>) -> T,
    {
        if self.all_true() {
            return f(&mut iter::repeat(true).take(self.len()));
        }

        if self.all_false() {
            return f(&mut iter::repeat(false).take(self.len()));
        }

        // We check for representations in order of performance, with BooleanBuffer iteration last.

        if let Some(indices) = self.0.maybe_indices() {
            let mut iter = IndicesBoolIter {
                indices: indices.iter().copied().peekable(),
                pos: 0,
                len: self.len(),
            };
            return f(&mut iter);
        }

        if let Some(slices) = self.0.maybe_slices() {
            let mut iter = SlicesBoolIter {
                slices: slices.iter().copied().peekable(),
                pos: 0,
                len: self.len(),
            };
            return f(&mut iter);
        }

        if let Some(buffer) = self.0.maybe_buffer() {
            return f(&mut buffer.iter());
        }

        unreachable!()
    }
}

struct IndicesBoolIter<I>
where
    I: Iterator<Item = usize>,
{
    indices: Peekable<I>,
    pos: usize,
    len: usize,
}

impl<I> Iterator for IndicesBoolIter<I>
where
    I: Iterator<Item = usize>,
{
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices.peek() {
            None => {
                if self.pos < self.len {
                    self.pos += 1;
                    return Some(false);
                }
                None
            }
            Some(next) => {
                if *next == self.pos {
                    self.indices.next();
                    self.pos += 1;
                    Some(true)
                } else {
                    self.pos += 1;
                    Some(false)
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.pos;
        (remaining, Some(remaining))
    }
}

unsafe impl<I: Iterator<Item = usize>> TrustedLen for IndicesBoolIter<I> {}

#[allow(dead_code)]
struct SlicesBoolIter<I>
where
    I: Iterator<Item = (usize, usize)>,
{
    slices: Peekable<I>,
    pos: usize,
    len: usize,
}

impl<I> Iterator for SlicesBoolIter<I>
where
    I: Iterator<Item = (usize, usize)>,
{
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let Some((start, end)) = self.slices.peek() else {
            if self.pos < self.len {
                self.pos += 1;
                return Some(false);
            }
            return None;
        };

        if self.pos < *start {
            self.pos += 1;
            return Some(false);
        }

        if self.pos == *end - 1 {
            self.slices.next();
        }

        self.pos += 1;
        Some(true)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.pos;
        (remaining, Some(remaining))
    }
}

unsafe impl<I: Iterator<Item = (usize, usize)>> TrustedLen for SlicesBoolIter<I> {}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::Mask;

    #[test]
    fn iter_bools_all_true() {
        let mask = Mask::new_true(10);
        assert_eq!(mask.iter_bools(|iter| iter.collect_vec()), vec![true; 10]);
    }

    #[test]
    fn iter_bools_all_false() {
        let mask = Mask::new_false(10);
        assert_eq!(mask.iter_bools(|iter| iter.collect_vec()), vec![false; 10]);
    }

    #[test]
    fn iter_bools_indices() {
        assert_eq!(
            Mask::from_indices(5, vec![]).iter_bools(|iter| iter.collect_vec()),
            vec![false; 5],
        );
        assert_eq!(
            Mask::from_indices(5, vec![0, 1, 2, 3, 4]).iter_bools(|iter| iter.collect_vec()),
            vec![true; 5],
        );
        assert_eq!(
            Mask::from_indices(5, vec![0, 4]).iter_bools(|iter| iter.collect_vec()),
            vec![true, false, false, false, true],
        );
        assert_eq!(
            Mask::from_indices(5, vec![1, 2, 3]).iter_bools(|iter| iter.collect_vec()),
            vec![false, true, true, true, false],
        );
    }

    #[test]
    fn iter_bools_slices() {
        assert_eq!(
            Mask::from_slices(5, vec![]).iter_bools(|iter| iter.collect_vec()),
            vec![false; 5],
        );
        assert_eq!(
            Mask::from_slices(5, vec![(0, 5)]).iter_bools(|iter| iter.collect_vec()),
            vec![true; 5],
        );
        assert_eq!(
            Mask::from_slices(5, vec![(0, 1), (4, 5)]).iter_bools(|iter| iter.collect_vec()),
            vec![true, false, false, false, true],
        );
        assert_eq!(
            Mask::from_slices(5, vec![(1, 4)]).iter_bools(|iter| iter.collect_vec()),
            vec![false, true, true, true, false],
        );
    }
}
