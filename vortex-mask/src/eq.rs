use crate::Mask;

impl PartialEq for Mask {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.true_count() != other.true_count() {
            return false;
        }

        // TODO(ngates): we could compare by indices if density is low enough
        self.boolean_buffer() == other.boolean_buffer()
    }
}

impl Eq for Mask {}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;

    use crate::Mask;

    #[test]
    fn filter_mask_eq() {
        assert_eq!(
            Mask::new_true(5),
            Mask::from_buffer(BooleanBuffer::new_set(5))
        );
        assert_eq!(
            Mask::new_false(5),
            Mask::from_buffer(BooleanBuffer::new_unset(5))
        );
        assert_eq!(
            Mask::from_indices(5, vec![0, 2, 3]),
            Mask::from_slices(5, vec![(0, 1), (2, 4)])
        );
        assert_eq!(
            Mask::from_indices(5, vec![0, 2, 3]),
            Mask::from_buffer(BooleanBuffer::from_iter([true, false, true, true, false]))
        );
    }
}
