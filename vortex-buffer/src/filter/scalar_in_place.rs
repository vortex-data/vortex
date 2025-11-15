// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::BitView;
use std::ptr;

/// Benchmark wrapper for [`filter_in_place_scalar`].
#[doc(hidden)]
#[cfg(feature = "bench")]
pub fn bench_filter_in_place_scalar<const NB: usize, T: Copy>(
    bit_view: &BitView<NB>,
    items: &mut [T],
) {
    filter_in_place_scalar(bit_view, items);
}

/// Filters the given slice of items in place according to the provided BitView using scalar
/// (non-SIMD) code.
///
/// The caller *should* handle where the BitView has zero or full true counts to avoid unnecessary
/// work.
pub(crate) fn filter_in_place_scalar<const NB: usize, T: Copy>(
    bit_view: &BitView<NB>,
    items: &mut [T],
) {
    let mut read_ptr = items.as_ptr();
    let mut write_ptr = items.as_mut_ptr();

    for mut word in bit_view.iter_words() {
        match word {
            usize::MAX => {
                // All items => copy usize::BITS items.
                unsafe {
                    // We cannot guarantee non-overlapping, so use ptr::copy rather than
                    // ptr::copy_nonoverlapping.
                    ptr::copy(read_ptr, write_ptr, usize::BITS as usize);
                    read_ptr = read_ptr.add(usize::BITS as usize);
                    write_ptr = write_ptr.add(usize::BITS as usize);
                }
            }
            _ => {
                // Note this also handles word == 0 case by skipping the loop entirely.
                while word != 0 {
                    let bit_pos = word.trailing_zeros();
                    unsafe {
                        ptr::copy(read_ptr.add(bit_pos as usize), write_ptr, 1);
                        write_ptr = write_ptr.add(1);
                    }
                    word &= word - 1; // Clear the bit at `bit_pos`
                }
                unsafe { read_ptr = read_ptr.add(usize::BITS as usize) };
            }
        }
    }
}
