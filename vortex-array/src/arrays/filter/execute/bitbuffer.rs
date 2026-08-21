// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`BitBuffer`] filtering algorithms.

use vortex_buffer::BitBuffer;
use vortex_mask::MaskValuesRef;

use crate::arrays::bool::compute::filter::filter_bitbuffer_by_mask;

/// Filter a [`BitBuffer`] by [`MaskValuesRef`], returning a new [`BitBuffer`].
pub(super) fn filter_bit_buffer(bb: &BitBuffer, mask: &MaskValuesRef) -> BitBuffer {
    assert_eq!(
        mask.len(),
        bb.len(),
        "Selection mask length must equal the mask length"
    );

    filter_bitbuffer_by_mask(bb, mask.bit_buffer(), mask.true_count())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;

    use super::filter_bit_buffer;

    #[test]
    fn filter_bool_by_mask_test() {
        let buf = bitbuffer![1 1 0];
        let mask = Mask::from_iter([true, false, true]);
        let Mask::Values(mask_values) = &mask else {
            panic!("a partially selective mask must contain mask values");
        };
        let filtered = filter_bit_buffer(&buf, mask_values);
        assert_eq!(2, filtered.len());
        assert_eq!(filtered, bitbuffer![1 0])
    }
}
