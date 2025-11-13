use vortex_mask::Mask;
use vortex_vector::null::{NullVector, NullVectorMut};

use crate::filter::{Filter, MaskIndices};

impl Filter<Mask> for &NullVector {
    type Output = NullVector;

    fn filter(self, selection: &Mask) -> Self::Output {
        NullVector::new(selection.true_count())
    }
}

impl Filter<MaskIndices<'_>> for &NullVector {
    type Output = NullVector;

    fn filter(self, indices: &MaskIndices) -> Self::Output {
        NullVector::new(indices.len())
    }
}

impl Filter<Mask> for &mut NullVectorMut {
    type Output = ();

    fn filter(self, selection: &Mask) -> Self::Output {
        *self = NullVectorMut::new(selection.true_count())
    }
}

impl Filter<MaskIndices<'_>> for &mut NullVectorMut {
    type Output = ();

    fn filter(self, indices: &MaskIndices) -> Self::Output {
        *self = NullVectorMut::new(indices.len())
    }
}
