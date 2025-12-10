// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::null::NullVector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &NullVector {
    type Output = NullVector;

    fn take(self, indices: &PVector<I>) -> NullVector {
        // NullVector is always all-null, so the result is just a new NullVector with the same
        // length as the indices. We don't need to check index validity since the result is all-null
        // regardless.
        NullVector::new(indices.len())
    }
}

impl<I: UnsignedPType> Take<[I]> for &NullVector {
    type Output = NullVector;

    fn take(self, indices: &[I]) -> NullVector {
        // NullVector is always all-null, so the result is just a new NullVector with the same
        // length as the indices.
        NullVector::new(indices.len())
    }
}
