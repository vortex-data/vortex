// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::decimal::DecimalVector;
use vortex_vector::match_each_dvector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &DecimalVector {
    type Output = DecimalVector;

    fn take(self, indices: &PVector<I>) -> DecimalVector {
        // If all the indices are valid, we can delegate to the slice indices implementation.
        if indices.validity().all_true() {
            return self.take(indices.elements().as_slice());
        }

        match_each_dvector!(self, |v| { v.take(indices).into() })
    }
}

impl<I: UnsignedPType> Take<[I]> for &DecimalVector {
    type Output = DecimalVector;

    fn take(self, indices: &[I]) -> DecimalVector {
        match_each_dvector!(self, |v| { v.take(indices).into() })
    }
}
