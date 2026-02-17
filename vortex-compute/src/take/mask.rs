// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Unsigned;
use vortex_mask::Mask;

use crate::take::Take;

impl<I: Unsigned + AsPrimitive<usize>> Take<[I]> for &Mask {
    type Output = Mask;

    fn take(self, indices: &[I]) -> Mask {
        match self {
            Mask::AllTrue(_) => Mask::AllTrue(indices.len()),
            Mask::AllFalse(_) => Mask::AllFalse(indices.len()),
            Mask::Values(mask_values) => {
                let taken_bit_buffer = mask_values.bit_buffer().take(indices);
                Mask::from_buffer(taken_bit_buffer)
            }
        }
    }
}
