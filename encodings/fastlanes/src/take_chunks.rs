// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take from encodings that decode in [`FL_CHUNK_SIZE`]-value chunks through a [`ChunkDecoder`].
//!
//! Callers check the indices with [`check_indices`] and turn them into positions with
//! [`valid_positions`], or [`nullable_positions`] when some are null, so that each case gets its own
//! loop.
//!
//! Both entry points decode the positions that fall in one chunk together: through
//! [`ChunkDecoder::decode_indices`] when there are at most `sparse_limit` of them, and otherwise by
//! decoding the whole chunk and picking the values out of it.
//!
//! - [`take_in_order`] visits the positions in the order given, so a chunk that the positions
//!   leave and come back to is decoded again. It suits decoders whose `decode_indices` is cheap.
//! - [`take_grouped`] first groups the positions by chunk, so each chunk is decoded at most once.
//!   It suits decoders that must decode a whole chunk to read any value of it.

use std::mem::MaybeUninit;
use std::slice::from_raw_parts;

use num_traits::AsPrimitive;
use vortex_array::arrays::filter::ChunkDecoder;
use vortex_array::dtype::IntegerPType;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::FL_CHUNK_SIZE;

/// Checks that every valid index, as `validity` says, lies within an array of `len` values.
pub(crate) fn check_indices<I: IntegerPType>(
    indices: &[I],
    validity: &Mask,
    len: usize,
) -> VortexResult<()> {
    // A negative index converts to a `usize` above any length, so one comparison checks both ends.
    let as_usize = |index: &I| AsPrimitive::<usize>::as_(*index);
    let invalid_index = match validity {
        // A plain maximum vectorizes, where a search for the first bad index would not.
        Mask::AllTrue(_) => indices.iter().map(as_usize).max().filter(|&max| max >= len),
        Mask::AllFalse(_) => None,
        Mask::Values(values) => indices
            .iter()
            .zip(values.bit_buffer().iter())
            .find(|&(index, valid)| valid && as_usize(index) >= len)
            .map(|(index, _)| as_usize(index)),
    };
    if let Some(index) = invalid_index {
        vortex_bail!("Take index {index} is out of bounds for length {len}");
    }
    Ok(())
}

/// The positions, counted from the start of the first chunk, of `indices` into an array that
/// starts `offset` into its first chunk, when every index is valid and checked.
pub(crate) fn valid_positions<I: IntegerPType>(
    indices: &[I],
    offset: usize,
) -> impl ExactSizeIterator<Item = usize> + Clone + '_ {
    indices
        .iter()
        .map(move |&index| offset + AsPrimitive::<usize>::as_(index))
}

/// Like [`valid_positions`], with null indices, as `validity` says, mapped to the array's first
/// value: the take reads it, and the taken validity masks it.
pub(crate) fn nullable_positions<'a, I: IntegerPType>(
    indices: &'a [I],
    validity: &'a Mask,
    offset: usize,
) -> impl ExactSizeIterator<Item = usize> + Clone + 'a {
    indices.iter().enumerate().map(move |(i, &index)| {
        if validity.value(i) {
            offset + AsPrimitive::<usize>::as_(index)
        } else {
            offset
        }
    })
}

/// Appends to `output` the values at `positions` in the order given.
///
/// `output` must have spare capacity for every position.
#[inline]
pub(crate) fn take_in_order<T: Copy>(
    decoder: &impl ChunkDecoder<T>,
    output: &mut BufferMut<T>,
    positions: impl Iterator<Item = usize>,
    sparse_limit: usize,
    scratch: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
) {
    chunked_positions(positions, |chunk, within| {
        if within.len() > sparse_limit {
            decoder.decode_chunk(chunk, scratch);
            output.extend_trusted(
                within
                    .iter()
                    // SAFETY: positions within a chunk are below `FL_CHUNK_SIZE`, and
                    // `decode_chunk` initialized every value of the chunk.
                    .map(|&index| unsafe { scratch.get_unchecked(index).assume_init() }),
            );
        } else {
            let start = output.len();
            decoder.decode_indices(
                chunk,
                within,
                &mut output.spare_capacity_mut()[..within.len()],
            );
            // SAFETY: `decode_indices` wrote one value for each position.
            unsafe { output.set_len(start + within.len()) };
        }
    });
}

/// The values at `positions`, which fall in the first `num_chunks` chunks, decoding each chunk at
/// most once.
pub(crate) fn take_grouped<T: Copy>(
    decoder: &impl ChunkDecoder<T>,
    positions: impl ExactSizeIterator<Item = usize> + Clone,
    num_chunks: usize,
    sparse_limit: usize,
) -> Buffer<T> {
    // Group the positions by chunk with a counting sort, keeping each one's output position.
    let mut chunk_starts = vec![0usize; num_chunks + 1];
    for position in positions.clone() {
        chunk_starts[position / FL_CHUNK_SIZE + 1] += 1;
    }
    for chunk in 0..num_chunks {
        chunk_starts[chunk + 1] += chunk_starts[chunk];
    }
    let len = positions.len();
    let mut next = chunk_starts.clone();
    let mut outputs = vec![0usize; len];
    let mut within = vec![0usize; len];
    for (out, position) in positions.enumerate() {
        let slot = &mut next[position / FL_CHUNK_SIZE];
        outputs[*slot] = out;
        within[*slot] = position % FL_CHUNK_SIZE;
        *slot += 1;
    }

    let mut output = BufferMut::<T>::with_capacity(len);
    let dst = &mut output.spare_capacity_mut()[..len];
    let mut values = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
    for chunk in 0..num_chunks {
        let group = chunk_starts[chunk]..chunk_starts[chunk + 1];
        if group.is_empty() {
            continue;
        }
        let within = &within[group.clone()];
        if within.len() > sparse_limit {
            decoder.decode_chunk(chunk, &mut values);
            for (&out, &index) in outputs[group].iter().zip(within) {
                dst[out] = values[index];
            }
        } else {
            let picked = &mut values[..within.len()];
            decoder.decode_indices(chunk, within, picked);
            for (&out, value) in outputs[group].iter().zip(picked.iter()) {
                dst[out] = *value;
            }
        }
    }
    // SAFETY: every position belongs to one chunk, so each output was written once above.
    unsafe { output.set_len(len) };
    output.freeze()
}

/// Groups consecutive `positions` by chunk without allocating, passing each run's chunk and the
/// positions within it to `chunk_fn`.
///
/// Repeated positions can select more than [`FL_CHUNK_SIZE`] values from one chunk, in which case
/// `chunk_fn` is called more than once for that chunk.
#[inline]
pub(crate) fn chunked_positions<F: FnMut(usize, &[usize])>(
    positions: impl Iterator<Item = usize>,
    mut chunk_fn: F,
) {
    let mut within_chunk = [const { MaybeUninit::<usize>::uninit() }; FL_CHUNK_SIZE];
    let mut len = 0;
    let mut current_chunk = 0;
    // SAFETY: Callers only read the first `len` entries, which have been written.
    let initialized =
        |buf: &[MaybeUninit<usize>], len| unsafe { from_raw_parts(buf.as_ptr().cast(), len) };

    for position in positions {
        let chunk = position / FL_CHUNK_SIZE;
        if len != 0 && (chunk != current_chunk || len == within_chunk.len()) {
            chunk_fn(current_chunk, initialized(&within_chunk, len));
            len = 0;
        }
        current_chunk = chunk;
        within_chunk[len].write(position % FL_CHUNK_SIZE);
        len += 1;
    }

    if len != 0 {
        chunk_fn(current_chunk, initialized(&within_chunk, len));
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::check_indices;
    use super::chunked_positions;
    use super::nullable_positions;
    use super::valid_positions;

    #[test]
    fn positions_map_nulls_to_the_first_value() -> VortexResult<()> {
        let indices = [3u32, 900, 7];
        let validity = Mask::from_iter([true, false, true]);
        // A null index may hold any value, even one outside the array.
        check_indices(&indices, &validity, 10)?;
        let actual: Vec<_> = nullable_positions(&indices, &validity, 100).collect();
        assert_eq!(actual, [103, 100, 107]);
        let all_valid: Vec<_> = valid_positions(&[3u32, 7], 100).collect();
        assert_eq!(all_valid, [103, 107]);
        Ok(())
    }

    #[test]
    fn check_indices_rejects_valid_indices_outside_the_array() {
        assert!(check_indices(&[3u32, 10], &Mask::new_true(2), 10).is_err());
        assert!(check_indices(&[-1i32], &Mask::new_true(1), 10).is_err());
        assert!(check_indices(&[3u32, 10], &Mask::from_iter([true, true]), 10).is_err());
    }

    #[test]
    fn chunked_positions_repeated() {
        let mut calls = Vec::new();
        chunked_positions([0; 1025].into_iter(), |chunk, within| {
            assert_eq!(chunk, 0);
            assert!(within.iter().all(|&position| position == 0));
            calls.push(within.len());
        });
        // More than 1024 positions in one chunk are split across calls.
        assert_eq!(calls, [1024, 1]);
    }

    #[test]
    fn chunked_positions_across_chunks() {
        let mut calls = Vec::new();
        chunked_positions([100, 105, 1100, 3100].into_iter(), |chunk, within| {
            calls.push((chunk, within.to_vec()));
        });
        assert_eq!(calls, [(0, vec![100, 105]), (1, vec![76]), (3, vec![28])]);
    }
}
