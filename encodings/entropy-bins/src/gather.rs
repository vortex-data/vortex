// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter and take that decode only the blocks holding a selected row.
//!
//! A sparse selection is gathered row by row in ascending order: consecutive selected blocks
//! decode together (four at a time in lockstep, like a full decompression), and a block selected
//! on its own decodes its ids only up to its last selected row, reading just the selected
//! offsets when few of its rows are selected. A denser filter decodes the selected blocks in
//! place and leaves the selection to the generic vectorized filter, so it is never slower than
//! decompressing first; a denser take touches nearly every block and is left to decompression.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::EntropyBins;
use crate::array::EntropyBinsData;
use crate::coder::CHUNK_VALUES;
use crate::decode::BlockView;
use crate::decode::OutInt;
use crate::decode::ids_slot;
use crate::decode::merge_block;
use crate::decode::parse_block;
use crate::decode::read_bits;

/// Filters selecting fewer rows than the array length over this are gathered row by row.
const FILTER_GATHER_DIVISOR: usize = 8;

/// Ascending takes of fewer rows than the array length over this are gathered row by row; larger
/// takes touch nearly every block, so they are left to decompression and the generic take.
const TAKE_GATHER_DIVISOR: usize = 64;

/// Unsorted takes of fewer rows than the array length over this are sorted and gathered.
const TAKE_SORT_DIVISOR: usize = 1024;

/// Blocks decoded together at most when gathering; bounds the scratch buffer.
const RUN_BLOCKS: usize = 16;

/// A lone block with fewer selected rows than its length over this reads their offsets directly
/// instead of merging the block.
const SPARSE_BLOCK_DIVISOR: usize = 32;

impl FilterKernel for EntropyBins {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Mask::Values(values) = mask else {
            return Ok(None);
        };
        let data = array.data();
        let validity = array.validity()?;
        if values.true_count() * FILTER_GATHER_DIVISOR < data.len() {
            let rows = values.indices();
            return match_each_integer_ptype!(data.ptype(), |T| {
                let mut out = BufferMut::<T>::with_capacity(rows.len());
                gather_sorted::<T>(data, rows, &mut out)?;
                Ok(Some(
                    PrimitiveArray::new(out, validity.filter(mask)?).into_array(),
                ))
            });
        }
        let bits = values.bit_buffer();
        let decoded = match_each_integer_ptype!(data.ptype(), |T| {
            let buffer = decode_touched::<T>(data, |lo, hi| bits.count_range(lo, hi) > 0)?;
            PrimitiveArray::new(buffer, validity)
        });
        Ok(Some(
            decoded
                .into_array()
                .filter(mask.clone())?
                .execute::<PrimitiveArray>(ctx)?
                .into_array(),
        ))
    }
}

impl TakeExecute for EntropyBins {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let data = array.data();
        let k = indices.len();
        if k * TAKE_GATHER_DIVISOR >= data.len() {
            return Ok(None);
        }
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let valid = indices.validity()?.execute_mask(indices.len(), ctx)?;
        let mut rows = match_each_integer_ptype!(indices.ptype(), |I| {
            valid_rows::<I>(indices.as_slice::<I>(), &valid, data.len())?
        });
        if !rows.is_sorted() {
            if k * TAKE_SORT_DIVISOR >= data.len() {
                return Ok(None);
            }
            rows.sort_unstable();
        }
        let validity = array.validity()?.take(&indices.into_array())?;
        match_each_integer_ptype!(data.ptype(), |T| {
            let out = take_rows::<T>(data, &rows, k)?;
            Ok(Some(PrimitiveArray::new(out, validity).into_array()))
        })
    }
}

fn in_bounds<I: IntegerPType>(index: I, len: usize) -> VortexResult<usize> {
    match index.to_usize().filter(|&row| row < len) {
        Some(row) => Ok(row),
        None => vortex_bail!("take index {index} out of bounds for length {len}"),
    }
}

/// Decode the array's rows, zeroing the blocks for which `touched` is false. `touched`
/// receives a block's rows relative to the slice, as `lo..hi`.
fn decode_touched<T: NativePType + OutInt>(
    data: &EntropyBinsData,
    touched: impl Fn(usize, usize) -> bool,
) -> VortexResult<Buffer<T>> {
    let (start, stop) = data.slice_range();
    if start == stop {
        return Ok(Buffer::empty());
    }
    let bv = data.block_values();
    let n_rows = data.unsliced_rows();
    let first = start / bv;
    let last = (stop - 1) / bv;
    let is_touched = |b: usize| {
        touched(
            (b * bv).max(start) - start,
            ((b + 1) * bv).min(stop) - start,
        )
    };
    let covered = ((last + 1) * bv).min(n_rows) - first * bv;
    let mut out = BufferMut::<T>::with_capacity(covered);
    // SAFETY: every block is either decoded or zeroed below.
    unsafe { out.set_len(covered) };
    let mut ids = vec![0u8; 4 * ids_slot(bv)];
    let mut b = first;
    while b <= last {
        if !is_touched(b) {
            let lo = (b - first) * bv;
            out[lo..(lo + bv).min(covered)].fill(T::default());
            b += 1;
            continue;
        }
        let mut end = b + 1;
        while end <= last && is_touched(end) {
            end += 1;
        }
        let lo = (b - first) * bv;
        let hi = (end * bv).min(n_rows) - first * bv;
        data.decode_blocks(b, end, &mut out[lo..hi], &mut ids)?;
        b = end;
    }
    let offset = start - first * bv;
    Ok(out.freeze().slice(offset..offset + (stop - start)))
}

/// The valid indices as `(row, output position)` pairs.
fn valid_rows<I: IntegerPType>(
    indices: &[I],
    valid: &Mask,
    len: usize,
) -> VortexResult<Vec<(usize, usize)>> {
    let mut rows = Vec::with_capacity(valid.true_count());
    for (pos, &index) in indices.iter().enumerate() {
        if valid.value(pos) {
            rows.push((in_bounds(index, len)?, pos));
        }
    }
    Ok(rows)
}

/// The values of the `(row, output position)` pairs (ascending by row), zero at the other
/// output positions.
fn take_rows<T: NativePType + OutInt>(
    data: &EntropyBinsData,
    rows: &[(usize, usize)],
    n_out: usize,
) -> VortexResult<BufferMut<T>> {
    let mut values = BufferMut::<T>::with_capacity(rows.len());
    let ascending: Vec<usize> = rows.iter().map(|&(row, _)| row).collect();
    gather_sorted::<T>(data, &ascending, &mut values)?;
    if rows.len() == n_out && rows.iter().enumerate().all(|(i, &(_, pos))| i == pos) {
        return Ok(values);
    }
    let mut out = BufferMut::<T>::zeroed(n_out);
    for (&(_, pos), &v) in rows.iter().zip(values.iter()) {
        out[pos] = v;
    }
    Ok(out)
}

/// Push the values of `rows` (ascending, relative to the array's slice) onto `out`.
fn gather_sorted<T: OutInt>(
    data: &EntropyBinsData,
    rows: &[usize],
    out: &mut BufferMut<T>,
) -> VortexResult<()> {
    let (start, _) = data.slice_range();
    let bv = data.block_values();
    let n_rows = data.unsliced_rows();
    let mut scratch = vec![T::default(); RUN_BLOCKS * bv];
    let mut ids = vec![0u8; 4 * ids_slot(bv)];
    let block_of = |i: usize| (start + rows[i]) / bv;
    let mut i = 0;
    while i < rows.len() {
        // The run of consecutive selected blocks from the next row's block.
        let first = block_of(i);
        let mut stop = first + 1;
        let mut j = i;
        loop {
            let stop_row = stop * bv - start;
            j += rows[j..].partition_point(|&r| r < stop_row);
            if j == rows.len() || stop - first == RUN_BLOCKS {
                break;
            }
            if block_of(j) != stop {
                break;
            }
            stop += 1;
        }
        let selected = &rows[i..j];
        if stop - first > 1 {
            let covered = (stop * bv).min(n_rows) - first * bv;
            data.decode_blocks(first, stop, &mut scratch[..covered], &mut ids)?;
            out.extend(selected.iter().map(|&r| scratch[r + start - first * bv]));
        } else {
            gather_block(data, first, selected, &mut scratch, &mut ids, out)?;
        }
        i = j;
    }
    Ok(())
}

/// Push the values of `selected` (rows relative to the slice, all in block `block`).
fn gather_block<T: OutInt>(
    data: &EntropyBinsData,
    block: usize,
    selected: &[usize],
    scratch: &mut [T],
    ids: &mut [u8],
    out: &mut BufferMut<T>,
) -> VortexResult<()> {
    let bv = data.block_values();
    let decoder = data.decoder(block * bv / CHUNK_VALUES)?;
    let block_len = bv.min(data.unsliced_rows() - block * bv);
    let view = parse_block(
        data.data.as_slice(),
        data.block_start(block),
        block_len,
        decoder.table.as_ref(),
    )?;
    // The position in the block of a row relative to the slice.
    let (start, _) = data.slice_range();
    let at = |r: usize| r + start - block * bv;
    let Some(&last) = selected.last() else {
        return Ok(());
    };
    let limit = at(last) + 1;
    decoder.ids(&view, &mut ids[..ids_slot(bv)], limit);
    let (seeds, lag) = data.seeds_of(block);
    if lag == 0 && selected.len() * SPARSE_BLOCK_DIVISOR < block_len {
        // Walk the offsets' bit position over the ids up to each selected row.
        let (mut walked, mut bit) = (0, 0);
        for &r in selected {
            let pos = at(r);
            while walked < pos {
                bit += decoder.widths[usize::from(ids[walked])] as usize;
                walked += 1;
            }
            let id = usize::from(ids[pos]);
            let offset = read_bits(view.offsets, bit, decoder.widths[id]);
            out.push(T::truncate_from(decoder.tl[id].wrapping_add(offset)));
        }
        return Ok(());
    }
    let prefix = BlockView { n: limit, ..view };
    merge_block(
        decoder,
        &prefix,
        &ids[..ids_slot(bv)],
        scratch,
        &seeds[..lag],
    );
    out.extend(selected.iter().map(|&r| scratch[at(r)]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::TakeExecute;
    use vortex_array::arrays::filter::FilterKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::BLOCK_VALUES;
    use crate::EntropyBins;
    use crate::EntropyBinsOptions;
    use crate::MAX_BLOCK_VALUES;
    use crate::tests::skewed;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    const N: usize = 20_000;

    fn arrays(options: EntropyBinsOptions) -> VortexResult<(ArrayRef, ArrayRef)> {
        let validity = Validity::from_iter((0..N).map(|i| i % 17 != 3));
        let prim = PrimitiveArray::new(Buffer::from(skewed(N, 7)), validity);
        let encoded = EntropyBins::from_primitive(prim.as_view(), 8, options)?.into_array();
        Ok((prim.into_array(), encoded))
    }

    /// Selections: very sparse, sparse runs across blocks, clustered, dense, and edge rows.
    fn selections(len: usize) -> Vec<Vec<usize>> {
        vec![
            vec![0],
            vec![len - 1],
            (0..len).step_by(997).collect(),
            (0..len).step_by(41).collect(),
            (0..len).filter(|i| (i / 300) % 5 == 1).collect(),
            (0..len).filter(|i| i % 3 != 0).collect(),
            (len / 2..len / 2 + 5000.min(len / 2)).collect(),
        ]
    }

    #[rstest]
    #[case(EntropyBinsOptions::new(0, BLOCK_VALUES))]
    #[case(EntropyBinsOptions::new(0, MAX_BLOCK_VALUES))]
    #[case(EntropyBinsOptions::new(2, BLOCK_VALUES))]
    #[case(EntropyBinsOptions::new(1, MAX_BLOCK_VALUES).with_word_bits(8))]
    fn filter_matches_primitive(#[case] options: EntropyBinsOptions) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (prim, encoded) = arrays(options)?;
        for (a, b) in [(0, N), (1500, 17_777)] {
            let (prim, encoded) = (prim.slice(a..b)?, encoded.slice(a..b)?);
            let view = encoded
                .as_opt::<EntropyBins>()
                .ok_or_else(|| vortex_err!("slice is not entropy bins"))?;
            for rows in selections(b - a) {
                let mask = Mask::from_indices(b - a, rows);
                let got = <EntropyBins as FilterKernel>::filter(view, &mask, &mut ctx)?
                    .ok_or_else(|| vortex_err!("kernel declined"))?;
                assert_arrays_eq!(got, prim.filter(mask)?, &mut ctx);
            }
        }
        Ok(())
    }

    #[rstest]
    #[case(EntropyBinsOptions::new(0, BLOCK_VALUES))]
    #[case(EntropyBinsOptions::new(3, 2 * BLOCK_VALUES))]
    fn take_matches_primitive(#[case] options: EntropyBinsOptions) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (prim, encoded) = arrays(options)?;
        let (prim, encoded) = (prim.slice(700..N)?, encoded.slice(700..N)?);
        let view = encoded
            .as_opt::<EntropyBins>()
            .ok_or_else(|| vortex_err!("slice is not entropy bins"))?;
        let len = N - 700;
        let mut takes: Vec<ArrayRef> = selections(len)
            .into_iter()
            .map(|rows| {
                PrimitiveArray::from_iter(rows.into_iter().map(|r| u32::try_from(r).unwrap_or(0)))
                    .into_array()
            })
            .collect();
        // Unsorted with duplicates, and nullable indices.
        takes.push(
            PrimitiveArray::from_iter([9000u64, 5, 9000, 18_000, 0, 1023, 1024]).into_array(),
        );
        takes.push(
            PrimitiveArray::from_option_iter([Some(3i16), None, Some(4000), None, Some(2)])
                .into_array(),
        );
        // Dense, unsorted and nullable: left to the generic take.
        takes.push(
            PrimitiveArray::from_option_iter(
                (0..3000u64)
                    .map(|i| (i % 7 != 0).then_some((i * 7919) % u64::try_from(len).unwrap_or(1))),
            )
            .into_array(),
        );
        let mut engaged = 0;
        for indices in takes {
            let want = prim.take(indices.clone())?;
            match <EntropyBins as TakeExecute>::take(view, &indices, &mut ctx)? {
                Some(got) => {
                    engaged += 1;
                    assert_arrays_eq!(got, want, &mut ctx);
                }
                None => assert_arrays_eq!(encoded.take(indices)?, want, &mut ctx),
            }
        }
        assert!(engaged >= 5, "kernel engaged {engaged} times");
        let len32 = u32::try_from(len).unwrap_or(0);
        let out_of_bounds = PrimitiveArray::from_iter([3, len32]).into_array();
        assert!(<EntropyBins as TakeExecute>::take(view, &out_of_bounds, &mut ctx).is_err());
        Ok(())
    }

    #[rstest]
    #[case(EntropyBinsOptions::new(0, BLOCK_VALUES))]
    #[case(EntropyBinsOptions::new(2, MAX_BLOCK_VALUES))]
    fn conformance(#[case] options: EntropyBinsOptions) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (_, encoded) = arrays(options)?;
        for array in [encoded.clone(), encoded.slice(5..3000)?] {
            test_filter_conformance(&array, &mut ctx);
            test_take_conformance(&array, &mut ctx);
        }
        Ok(())
    }
}
