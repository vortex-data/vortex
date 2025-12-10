// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::match_each_pvector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &PrimitiveVector {
    type Output = PrimitiveVector;

    fn take(self, indices: &PVector<I>) -> PrimitiveVector {
        // If all the indices are valid, we can delegate to the slice indices implementation.
        if indices.validity().all_true() {
            return self.take(indices.elements().as_slice());
        }

        match_each_pvector!(self, |v| { v.take(indices).into() })
    }
}

impl<I: UnsignedPType> Take<[I]> for &PrimitiveVector {
    type Output = PrimitiveVector;

    fn take(self, indices: &[I]) -> PrimitiveVector {
        match_each_pvector!(self, |v| { v.take(indices).into() })
    }
}
