// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visits a slice one 64-lane chunk at a time, together with the chunk's validity word.

use vortex_buffer::BitBuffer;

/// Calls `f` with each chunk of 64 `values` and its validity word, where bit `i` is set if
/// `chunk[i]` is valid. A `mask` of `None` means every value is valid.
///
/// The mask is read one word per chunk, so `f` can skip a word of zeros and take a branch-free
/// path for a full word without testing each bit. The trailing values are padded into a last
/// chunk whose padding bits are unset.
///
/// # Panics
///
/// Panics if `mask` and `values` have different lengths.
#[inline]
pub fn for_each_chunk<T, F>(values: &[T], mask: Option<&BitBuffer>, mut f: F)
where
    T: Copy,
    F: FnMut(&[T; 64], u64),
{
    let (chunks, remainder) = values.as_chunks::<64>();
    let remainder_bits = match mask {
        None => {
            chunks.iter().for_each(|chunk| f(chunk, u64::MAX));
            (1u64 << remainder.len()) - 1
        }
        Some(mask) => {
            assert_eq!(
                values.len(),
                mask.len(),
                "values and mask must have the same length"
            );
            let words = mask.chunks();
            chunks
                .iter()
                .zip(words.iter())
                .for_each(|(chunk, word)| f(chunk, word));
            words.remainder_bits()
        }
    };

    if let Some(&pad) = remainder.first() {
        let mut last = [pad; 64];
        last[..remainder.len()].copy_from_slice(remainder);
        f(&last, remainder_bits);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;

    use super::for_each_chunk;

    #[rstest]
    fn matches_mask_bits(
        #[values(0, 1, 63, 64, 65, 200)] len: usize,
        #[values(0, 3)] offset: usize,
        #[values(false, true)] all_valid: bool,
    ) {
        let values: Vec<usize> = (0..len).collect();
        // Slicing at `offset` covers words that do not start at a byte boundary.
        let mask = BitBuffer::from_iter((0..len + offset).map(|i| i % 3 != 0 && i % 7 != 0))
            .slice(offset..offset + len);
        let mask = (!all_valid).then_some(&mask);

        let mut chunks = 0;
        let mut valid = Vec::new();
        for_each_chunk(&values, mask, |chunk, word| {
            for (i, &value) in chunk.iter().enumerate() {
                if word & (1 << i) != 0 {
                    valid.push(value);
                }
            }
            chunks += 1;
        });

        assert_eq!(chunks, len.div_ceil(64));
        let expected: Vec<usize> = (0..len)
            .filter(|&i| mask.is_none_or(|m| m.value(i)))
            .collect();
        assert_eq!(valid, expected);
    }
}
