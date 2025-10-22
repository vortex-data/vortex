// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute::vectors::filter::Filter;
use vortex_buffer::BitBufferMut;
use vortex_mask::{Mask, MaskMut};

impl Filter for Mask {
    type Mutable = MaskMut;

    fn filter(&self, mask: &Mask) -> Self {
        self.filter_into(mask, MaskMut::empty())
    }

    fn filter_into(&self, mask: &Mask, out: Self::Mutable) -> Self {
        assert_eq!(self.len(), mask.len());

        match (self, mask) {
            (Mask::AllTrue(_), _) => Mask::AllTrue(mask.true_count()),
            (Mask::AllFalse(_), _) => Mask::AllFalse(mask.true_count()),

            (Mask::Values(_), Mask::AllTrue(_)) => self.clone(),
            (Mask::Values(_), Mask::AllFalse(_)) => {
                assert!(out.is_empty());
                out.freeze()
            }
            (Mask::Values(v1), Mask::Values(_)) => {
                Mask::from(v1.bit_buffer().filter_into(mask, BitBufferMut::empty()))
            }
        }
    }
}
