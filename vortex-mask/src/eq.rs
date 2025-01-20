use crate::Mask;

impl PartialEq for Mask {
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
