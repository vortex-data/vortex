use std::ops::BitAnd;

use vortex_error::vortex_panic;

use crate::Mask;

impl BitAnd for &Mask {
    type Output = Mask;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }
        if self.true_count() == 0 || rhs.true_count() == 0 {
            return Mask::new_false(self.len());
        }
        if self.true_count() == self.len() {
            return rhs.clone();
        }
        if rhs.true_count() == self.len() {
            return self.clone();
        }

        if let (Some(lhs), Some(rhs)) = (self.0.buffer.get(), rhs.0.buffer.get()) {
            return Mask::from_buffer(lhs & rhs);
        }

        if let (Some(lhs), Some(rhs)) = (self.0.indices.get(), rhs.0.indices.get()) {
            // TODO(ngates): this may only make sense for sparse indices.
            return Mask::from_intersection_indices(
                self.len(),
                lhs.iter().copied(),
                rhs.iter().copied(),
            );
        }

        // TODO(ngates): we could perform a more efficient bitandion for slices.
        Mask::from_buffer(self.boolean_buffer() & rhs.boolean_buffer())
    }
}
