// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A naive walk over unset bits, which [`BitBuffer`] only offers for set bits.

use vortex_buffer::BitBuffer;

/// Call `f` with the index of every unset bit, in ascending order.
pub(super) fn for_each_unset_index(bits: &BitBuffer, mut f: impl FnMut(usize)) {
    let len = bits.len();
    let mut start = 0;
    // The remainder word comes last, zero padded even when it holds no bits, so its padding comes
    // out set once inverted and has to be masked off.
    for chunk in bits.chunks().iter_padded() {
        let held = len - start;
        let mut word = if held < 64 {
            !chunk & ((1 << held) - 1)
        } else {
            !chunk
        };
        while word != 0 {
            f(start + word.trailing_zeros() as usize);
            word &= word - 1;
        }
        start += 64;
    }
}
