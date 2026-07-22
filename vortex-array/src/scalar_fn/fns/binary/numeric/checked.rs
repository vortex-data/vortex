// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared checked-lane execution helpers for numeric kernels.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

/// The values produced by a checked lane loop, plus whether any lane failed.
pub(super) struct CheckedValues<T> {
    pub(super) values: Buffer<T>,
    pub(super) failed: bool,
}

impl<T> CheckedValues<T> {
    pub(super) fn zeroed(len: usize) -> Self {
        Self {
            values: Buffer::<T>::zeroed(len),
            failed: false,
        }
    }

    fn failed(len: usize) -> Self {
        Self {
            values: Buffer::<T>::zeroed(len),
            failed: true,
        }
    }
}

// Checked one-pass ops delay early exit until the end of a small block. This
// keeps the loop generic while avoiding a branch-driven exit decision on every
// lane; it is deliberately independent of mask density or input length.
const CHECKED_BLOCK_LANES: usize = 16;

pub(super) fn checked_all_lanes<T, F>(len: usize, mut checked_at: F) -> CheckedValues<T>
where
    T: Default,
    F: FnMut(usize) -> Option<T>,
{
    let mut values = BufferMut::<T>::with_capacity(len);
    let mut base = 0;

    while base + CHECKED_BLOCK_LANES <= len {
        let mut block_failed = false;
        for idx in base..base + CHECKED_BLOCK_LANES {
            match checked_at(idx) {
                Some(value) => {
                    // SAFETY: the buffer is allocated with capacity `len`, and
                    // this loop pushes at most one value for each `idx`.
                    unsafe { values.push_unchecked(value) };
                }
                None => {
                    block_failed = true;
                    // SAFETY: the buffer is allocated with capacity `len`, and
                    // this loop pushes at most one value for each `idx`.
                    unsafe { values.push_unchecked(T::default()) };
                }
            }
        }

        if block_failed {
            return CheckedValues::failed(len);
        }
        base += CHECKED_BLOCK_LANES;
    }

    for idx in base..len {
        let Some(value) = checked_at(idx) else {
            return CheckedValues::failed(len);
        };
        // SAFETY: the buffer is allocated with capacity `len`, and this loop
        // pushes at most one value for each `idx`.
        unsafe { values.push_unchecked(value) };
    }

    CheckedValues {
        values: values.freeze(),
        failed: false,
    }
}

pub(super) fn checked_valid_lanes<T, F>(
    len: usize,
    valid_bits: &BitBuffer,
    mut checked_at: F,
) -> CheckedValues<T>
where
    T: Default,
    F: FnMut(usize) -> Option<T>,
{
    let mut values = BufferMut::<T>::zeroed(len);
    let mut failed = false;
    {
        let values = values.as_mut_slice();
        for_each_valid_idx(len, valid_bits, |idx| {
            let Some(value) = checked_at(idx) else {
                failed = true;
                return false;
            };
            values[idx] = value;
            true
        });
    }

    CheckedValues {
        values: values.freeze(),
        failed,
    }
}

pub(super) fn any_valid_error<F>(len: usize, valid_bits: &BitBuffer, is_error: F) -> bool
where
    F: Fn(usize) -> bool,
{
    !for_each_valid_idx(len, valid_bits, |idx| !is_error(idx))
}

fn for_each_valid_idx<F>(len: usize, valid_bits: &BitBuffer, mut f: F) -> bool
where
    F: FnMut(usize) -> bool,
{
    debug_assert_eq!(len, valid_bits.len());

    for (word_idx, valid_word) in valid_bits.chunks().iter_padded().enumerate() {
        if valid_word == 0 {
            continue;
        }

        let offset = word_idx * 64;
        let lanes = len.saturating_sub(offset).min(64);
        if lanes == 64 && valid_word == u64::MAX {
            for bit_idx in 0..64 {
                if !f(offset + bit_idx) {
                    return false;
                }
            }
            continue;
        }

        let mut valid_word = if lanes == 64 {
            valid_word
        } else {
            valid_word & ((1u64 << lanes) - 1)
        };
        while valid_word != 0 {
            let bit_idx = valid_word.trailing_zeros() as usize;
            if !f(offset + bit_idx) {
                return false;
            }
            valid_word &= valid_word - 1;
        }
    }

    true
}
