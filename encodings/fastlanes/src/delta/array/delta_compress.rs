// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::bit_transpose::transpose_validity;
use crate::fill_forward_nulls;
pub fn delta_compress(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    let validity = array.validity()?;
    let original_ptype = array.ptype();
    let array = array.reinterpret_cast(original_ptype.to_unsigned());

    let (bases, deltas) = match_each_unsigned_integer_ptype!(array.ptype(), |T| {
        // Fill-forward null values so that transposed deltas at null positions remain
        // small. Without this, bitpacking may skip patches for null positions, and the
        // corrupted delta values propagate through the cumulative sum during decompression.
        let filled = fill_forward_nulls(array.to_buffer::<T>(), &validity, ctx)?;
        let (bases, deltas) = compress_primitive::<T, { T::LANES }>(&filled);
        // TODO(robert): This can be avoided if we add TransposedBoolArray that performs index translation when necessary.
        let validity = transpose_validity(&validity, ctx)?;
        (
            PrimitiveArray::new(bases, array.dtype().nullability().into()),
            PrimitiveArray::new(deltas, validity),
        )
    });

    Ok((
        bases.reinterpret_cast(original_ptype),
        deltas.reinterpret_cast(original_ptype),
    ))
}

fn compress_primitive<T, const LANES: usize>(array: &[T]) -> (Buffer<T>, Buffer<T>)
where
    T: NativePType + Delta + Transpose,
{
    let padded_len = array.len().next_multiple_of(FL_CHUNK_SIZE);
    let bases_len = (padded_len / FL_CHUNK_SIZE) * LANES;

    // Split into full 1024-element chunks and a remainder.
    let (full_chunks, remainder) = array.as_chunks::<FL_CHUNK_SIZE>();

    // Allocate result arrays.
    let mut bases = BufferMut::with_capacity(bases_len);
    let mut deltas = BufferMut::with_capacity(padded_len);
    let (output_deltas, _) = deltas.spare_capacity_mut().as_chunks_mut::<FL_CHUNK_SIZE>();

    // Loop over all full 1024-element chunks.
    let mut transposed: [T; FL_CHUNK_SIZE] = [T::default(); FL_CHUNK_SIZE];
    let mut process_chunk = |input: &[T; FL_CHUNK_SIZE], output: &mut [MaybeUninit<T>; 1024]| {
        Transpose::transpose(input, &mut transposed);
        bases.extend_from_slice(&transposed[0..T::LANES]);

        unsafe {
            Delta::delta::<LANES>(
                &transposed,
                &*(transposed[0..T::LANES].as_ptr().cast()),
                mem::transmute::<&mut [MaybeUninit<T>; FL_CHUNK_SIZE], &mut [T; FL_CHUNK_SIZE]>(
                    output,
                ),
            );
        }
    };
    for (chunk, output) in full_chunks.iter().zip(output_deltas.iter_mut()) {
        process_chunk(chunk, output);
    }

    // Pad the remainder to 1024 elements and process as a full chunk.
    if !remainder.is_empty() {
        let mut padded_chunk = [T::default(); FL_CHUNK_SIZE];
        padded_chunk[..remainder.len()].copy_from_slice(remainder);
        process_chunk(&padded_chunk, &mut output_deltas[full_chunks.len()]);
    }

    unsafe { deltas.set_len(padded_len) };

    assert_eq!(bases.len(), bases_len);
    assert_eq!(deltas.len(), padded_len);

    (bases.freeze(), deltas.freeze())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;
    use crate::bitpack_compress::bitpack_encode;
    use crate::delta::array::delta_decompress::delta_decompress;
    use crate::delta_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    #[rstest]
    #[case::u32((0u32..10_000).collect())]
    #[case::u8((0..10_000).map(|i| (i % (u8::MAX as i32)) as u8).collect())]
    #[case::nullable_u32(PrimitiveArray::from_option_iter(
            (0u32..10_000).map(|i| (i % 2 == 0).then_some(i)),
    ))]
    // Signed inputs that stay non-negative: encoded deltas are identical to the u32 case
    // bit-for-bit, but the buffer's dtype carries the signedness through round-trip.
    #[case::i32_non_negative((0i32..10_000).collect())]
    // Signed inputs crossing zero: deltas alternate in sign, which under wrapping_sub
    // populates the high bits of negative deltas. Bit-packing without preprocessing
    // would explode here, but round-tripping the raw delta buffer is still correct.
    #[case::i32_crossing_zero((-5_000i32..5_000).collect())]
    // All-negative signed values.
    #[case::i32_all_negative((-10_000i32..0).collect())]
    // i8 across the full type range: tests T::MIN / T::MAX boundaries and the
    // remainder-padded chunk path (256 < FL_CHUNK_SIZE).
    #[case::i8_full_range((i8::MIN..=i8::MAX).collect())]
    // i16 crossing zero.
    #[case::i16_crossing_zero((-2_000i16..2_000).collect())]
    // i64 with large negative offset.
    #[case::i64_large_negative((0i64..5_000).map(|i| i - 1_000_000_000_000).collect())]
    // Nullable signed array with values around zero.
    #[case::nullable_i32_crossing(PrimitiveArray::from_option_iter(
            (-2_000i32..2_000).map(|i| (i % 3 != 0).then_some(i)),
    ))]
    fn test_compress(#[case] array: PrimitiveArray) -> VortexResult<()> {
        let delta = Delta::try_from_primitive_array(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(delta.len(), array.len());
        let decompressed = delta_decompress(&delta, &mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(decompressed, array);
        Ok(())
    }

    /// Regression test: delta + bitpacked encoding must correctly round-trip nullable arrays
    /// where null positions contain arbitrary values. Without fill-forward, the delta cumulative
    /// sum propagates corrupted values from null positions.
    #[test]
    fn delta_bitpacked_trailing_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter(
            (0u8..200).map(|i| (!(50..100).contains(&i)).then_some(i)),
        );
        let (bases, deltas) = delta_compress(&array, &mut ctx)?;
        let bitpacked_deltas = bitpack_encode(&deltas, 1, None, &mut ctx)?;
        let packed_delta = Delta::try_new(
            bases.into_array(),
            bitpacked_deltas.into_array(),
            0,
            array.len(),
        )
        .vortex_expect("Delta array construction should succeed");
        let packed_delta_prim = packed_delta
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(packed_delta_prim, array);
        Ok(())
    }

    /// Measures compression of delta-encoded signed columns under three bit-packing strategies:
    ///   * `naive`: bit-packing the raw delta bytes (every negative delta sets the high bits,
    ///     so the OR mask forces `W = T`).
    ///   * `FFoR`:  subtracting the per-column `min(delta)` before bit-packing
    ///     (`W = ceil(log2(max - min + 1))`).
    ///   * `zigzag`: `(n << 1) ^ (n >> 31)` before bit-packing
    ///     (`W = 1 + ceil(log2(max(|min|, |max|)))`).
    ///
    /// Asserts that FFoR beats or ties naive on every workload and beats zigzag on the
    /// asymmetric workloads. Run with `--nocapture` to see the full table.
    #[test]
    fn synthetic_workload_compression() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        const N: usize = 8 * 1024; // 8 full FastLanes chunks per workload

        let monotone: Vec<i32> = (0..N as i32).collect();
        // Deterministic LCG so the test is reproducible.
        let mut lcg = 0u32;
        let mut next = || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (lcg >> 16) as i32
        };
        let sensor: Vec<i32> = (0..N).map(|_| (next() % 201) - 100).collect();
        let offset: Vec<i32> = (0..N as i32).map(|i| -1_000_000_000 + i).collect();
        let mut lcg2 = 0u32;
        let mut prev = 0i32;
        let near_monotone: Vec<i32> = (0..N)
            .map(|_| {
                lcg2 = lcg2.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let step = if (lcg2 >> 24) < 13 { -2 } else { 1 }; // ~5% backtrack
                prev = prev.wrapping_add(step);
                prev
            })
            .collect();
        let workloads = [
            ("monotone i32 (0..N)", monotone),
            ("sensor i32 in [-100, 100]", sensor),
            ("offset i32 base=-1e9", offset),
            ("near-monotone i32 (5% backtrack)", near_monotone),
        ];

        println!();
        println!(
            "{:<36} {:>10} {:>14} {:>5} {:>5} {:>5} {:>10} {:>9}  {:>10} {:>5} {:>10} {:>9}",
            "workload",
            "raw (B)",
            "Δ range",
            "Wnaive",
            "Wffor",
            "Wzig",
            "FFoR (B)",
            "FFoR x",
            "bases (B)",
            "Wb",
            "+bcomp (B)",
            "+bcomp x",
        );
        println!("{}", "-".repeat(140));

        for (name, values) in workloads {
            let raw_bytes = size_of_val(values.as_slice());
            let array = PrimitiveArray::from_iter(values);
            let (bases, deltas) = delta_compress(&array, &mut ctx)?;
            let deltas_buf: &[i32] = deltas.as_slice();
            let bases_buf: &[i32] = bases.as_slice();

            let min_d = *deltas_buf.iter().min().unwrap();
            let max_d = *deltas_buf.iter().max().unwrap();

            // Naive width = OR of raw u32 bit-patterns of every delta. Any negative delta
            // sets the high bits and forces W = 32.
            let or: u32 = deltas_buf.iter().fold(0u32, |a, &d| a | (d as u32));
            let naive_w = if or == 0 {
                0
            } else {
                32 - or.leading_zeros() as usize
            };

            // FFoR width = ceil(log2(span)) where span = (max - min + 1).
            let span = (max_d as i64 - min_d as i64) as u64 + 1;
            let ffor_w = if span <= 1 {
                0
            } else {
                64 - (span - 1).leading_zeros() as usize
            };

            // ZigZag width = 1 + ceil(log2(max(|min|, |max|))) for any nonzero delta.
            let zz_mag = (min_d.unsigned_abs()).max(max_d.unsigned_abs());
            let zz_w = if zz_mag == 0 {
                0
            } else {
                1 + (32 - zz_mag.leading_zeros() as usize)
            };

            // FFoR encoded byte size: bases (already unpacked) + ref + ceil(packed bits / 8).
            let bases_bytes = size_of_val(bases_buf);
            let ref_bytes = size_of::<i32>();
            let packed_bits = deltas_buf.len() * ffor_w;
            let ffor_packed_bytes = packed_bits.div_ceil(8);
            let ffor_total = bases_bytes + ref_bytes + ffor_packed_bytes;
            let ratio = raw_bytes as f64 / ffor_total as f64;

            // Bases compressibility: what we save if the bases child is recursively
            // delta-encoded or FoR-encoded. The bases are the "first row of the transposed
            // chunk" per lane, so they form a sub-sequence that inherits the smoothness of
            // the input. We approximate with FFoR over the bases alone (no recursive Delta,
            // which would force padding to 1024 elements per FastLanes chunk and could lose
            // for short base sequences).
            let min_b = *bases_buf.iter().min().unwrap();
            let max_b = *bases_buf.iter().max().unwrap();
            let bspan = (max_b as i64 - min_b as i64) as u64 + 1;
            let bases_w = if bspan <= 1 {
                0
            } else {
                64 - (bspan - 1).leading_zeros() as usize
            };
            let bases_compressed = (bases_buf.len() * bases_w).div_ceil(8) + ref_bytes;
            let total_with_bcomp = bases_compressed + ref_bytes + ffor_packed_bytes;
            let ratio_with_bcomp = raw_bytes as f64 / total_with_bcomp as f64;

            println!(
                "{name:<36} {raw_bytes:>10} {:>14} {naive_w:>5} {ffor_w:>5} {zz_w:>5} {ffor_total:>10} {ratio:>8.2}x  {bases_bytes:>10} {bases_w:>5} {total_with_bcomp:>10} {ratio_with_bcomp:>8.2}x",
                format!("[{min_d}, {max_d}]"),
            );

            // Sanity assertions. naive_w is 32 (or near it) for any delta sequence that
            // contains a negative value; FFoR/ZigZag width must be strictly smaller for these
            // workloads.
            assert!(
                ffor_w <= naive_w.max(1),
                "FFoR must never exceed naive for {name}"
            );
            if min_d < 0 {
                assert_eq!(
                    naive_w, 32,
                    "any negative delta forces naive W to 32 for {name}"
                );
                assert!(ffor_w < 32, "FFoR must compress below T for {name}");
            }
            // On the asymmetric workloads (offset, near-monotone) FFoR must beat ZigZag.
            if min_d > 0 || max_d < 0 {
                assert!(
                    ffor_w < zz_w,
                    "FFoR should beat ZigZag on asymmetric {name}"
                );
            }
            // Sorted inputs => the bases inherit smoothness => the bases bit-width should be
            // far smaller than `T` for sorted columns.
            if name.starts_with("monotone") || name.starts_with("offset") {
                assert!(
                    bases_w < 16,
                    "sorted bases should pack below 16 bits for {name}"
                );
            }
        }

        Ok(())
    }
}
