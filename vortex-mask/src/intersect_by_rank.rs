use crate::Mask;

impl Mask {
    /// Take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// We are more interested in low selectivity `self` (as indices) with a boolean buffer mask,
    /// so we don't optimize for other cases, yet.
    ///
    /// # Examples
    ///
    /// Keep the third and fifth set values from mask `m1`:
    /// ```
    /// use vortex_mask::Mask;
    ///
    /// let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
    /// let m2 = Mask::from_iter([false, false, true, false, true]);
    /// assert_eq!(
    ///     m1.intersect_by_rank(&m2),
    ///     Mask::from_iter([false, false, false, false, true, false, false, true])
    /// );
    /// ```
    pub fn intersect_by_rank(&self, mask: &Mask) -> Mask {
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

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;

    use crate::Mask;

    #[test]
    fn mask_bitand_all_as_bit_and() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![true, true, true, true, true]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            false, true, false, true, true,
        ]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![1, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_all_true() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            false, false, true, true, true,
        ]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![true, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![2, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_true() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            true, false, false, true, true,
        ]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![true, false, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![0, 4])
        );
    }

    #[test]
    fn mask_bitand_false() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            true, false, false, true, true,
        ]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![false, false, false]));
        assert_eq!(this.intersect_by_rank(&mask), Mask::from_indices(5, vec![]));
    }
}
