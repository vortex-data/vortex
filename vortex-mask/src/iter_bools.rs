use std::iter;

use crate::{AllOr, Mask};

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
        match self.boolean_buffer() {
            AllOr::All => f(&mut iter::repeat_n(true, self.len())),
            AllOr::None => f(&mut iter::repeat_n(false, self.len())),
            AllOr::Some(buffer) => f(&mut buffer.iter()),
        }
    }
}

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
