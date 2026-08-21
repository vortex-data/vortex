// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::cast::test_cast_conformance;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::not_eq;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::scalar::Scalar;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::EliasFano;
use crate::EliasFanoArray;
use crate::EliasFanoArraySlotsExt;
use crate::EliasFanoCursor;
use crate::EliasFanoData;
use crate::elias_fano_encode;
use crate::params;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    crate::initialize(&session);
    session
});

/// Deterministic xorshift, so a failure is reproducible without a `rand` dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// Uniform over `0..=bound`, including when `bound` is `u64::MAX` and `bound + 1` overflows.
    fn at_most(&mut self, bound: u64) -> u64 {
        match bound.checked_add(1) {
            Some(universe) => self.below(universe),
            None => self.next_u64(),
        }
    }

    /// A random permutation of `0..len`, so index-order bugs cannot hide behind a sequential walk.
    fn permutation(&mut self, len: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..len).collect();
        for i in (1..len).rev() {
            indices.swap(i, self.below(i as u64 + 1) as usize);
        }
        indices
    }
}

/// A sorted sequence of `n` values spread over `0..=span`, with duplicates wherever they fall.
fn sorted_values(n: usize, span: u64, seed: u64) -> Vec<u64> {
    let mut rng = Rng(seed);
    let mut values: Vec<u64> = (0..n).map(|_| rng.at_most(span)).collect();
    values.sort_unstable();
    values
}

fn encode<P: NativePType>(values: &[P]) -> VortexResult<EliasFanoArray> {
    let array = PrimitiveArray::from_iter(values.iter().copied());
    let mut ctx = SESSION.create_execution_ctx();
    elias_fano_encode(array.as_ref().as_::<Primitive>(), &mut ctx)
}

/// Every element, read back through the cursor in a random order, must match `expected`.
fn check_access(array: &EliasFanoArray, expected: &[Scalar], seed: u64) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut cursor = EliasFanoCursor::try_new(array.as_view(), &mut ctx)?;
    for index in Rng(seed).permutation(expected.len()) {
        assert_eq!(cursor.access(index)?, expected[index], "index {index}");
    }
    // And once more in order, which takes the cursor's step-forward path instead of a select.
    let mut cursor = EliasFanoCursor::try_new(array.as_view(), &mut ctx)?;
    for (index, want) in expected.iter().enumerate() {
        assert_eq!(&cursor.access(index)?, want, "sequential index {index}");
    }
    // And through `scalar_at`, which builds no cursor at all and so shares only the layout with the
    // two loops above. Every shape the suite covers therefore checks the two readers against each
    // other, including the low-bits children a rewrite can leave behind. Order is irrelevant here:
    // the path is stateless.
    for (index, want) in expected.iter().enumerate() {
        assert_eq!(
            &array.execute_scalar(index, &mut ctx)?,
            want,
            "scalar_at index {index}"
        );
    }
    Ok(())
}

/// `next_geq`, `rank`, and `rank_inclusive` must agree with a linear scan, for probes on and off
/// the elements.
///
/// All three share one cursor, so they interleave the way a query does: `rank_inclusive` probes the
/// successor of the value `next_geq` just answered for, which is where a stale seat or a stale
/// memoised answer would show up.
fn check_searches(
    array: &EliasFanoArray,
    expected: &[Scalar],
    probes: &[Scalar],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut cursor = EliasFanoCursor::try_new(array.as_view(), &mut ctx)?;
    for probe in probes {
        let want_rank = expected.iter().take_while(|value| *value < probe).count();
        let want_value = expected.get(want_rank).cloned();

        let (rank, value) = cursor.next_geq(probe)?;
        assert_eq!(rank, want_rank, "rank of {probe}");
        assert_eq!(value, want_value, "next_geq of {probe}");
        // Repeat, which takes the memoised answer. It has to reproduce the whole answer, not just
        // the rank: the element found is generally not the probe.
        assert_eq!(
            cursor.next_geq(probe)?,
            (want_rank, want_value),
            "memoised next_geq of {probe}"
        );
        // The other end of the run equal to the probe, which together with `rank` brackets it.
        let want_inclusive = expected.iter().take_while(|value| *value <= probe).count();
        assert_eq!(
            cursor.rank_inclusive(probe)?,
            want_inclusive,
            "rank_inclusive of {probe}"
        );
    }
    Ok(())
}

/// Probes covering every element, both neighbours of every element, and the universe edges.
fn probes(values: &[u64], span: u64, dtype: &DType, seed: u64) -> Vec<Scalar> {
    let ptype = dtype.as_ptype();
    let mut raw: Vec<u64> = Vec::new();
    for &value in values {
        raw.push(value);
        raw.push(value.saturating_sub(1));
        raw.push(value.saturating_add(1).min(span));
    }
    raw.push(0);
    raw.push(span);
    let mut rng = Rng(seed);
    raw.extend((0..64).map(|_| rng.at_most(span)));
    // Shuffled, so the cursor has to reseat backwards as well as walk forwards.
    for i in (1..raw.len()).rev() {
        raw.swap(i, rng.below(i as u64 + 1) as usize);
    }
    raw.into_iter()
        .map(|value| scalar_of(ptype, value))
        .collect()
}

/// A sorted sequence of `clusters` tight runs, each `per_cluster` long, over `0..=span`.
///
/// `lower_width` is picked for a uniform spread, so uniformly random values leave about one element
/// in each high-part bucket however wide the universe is. Clustering is what puts many elements in
/// one bucket, which is the shape a search inside a bucket has to handle without walking it.
fn clustered_values(clusters: usize, per_cluster: usize, span: u64, seed: u64) -> Vec<u64> {
    let mut rng = Rng(seed);
    let mut values: Vec<u64> = Vec::with_capacity(clusters * per_cluster);
    for _ in 0..clusters {
        let base = rng.at_most(span);
        for _ in 0..per_cluster {
            // A handful of distinct values per run, so the low bits inside a bucket vary rather
            // than every comparison in it landing on the same answer.
            values.push(base.saturating_add(rng.below(4)).min(span));
        }
    }
    values.sort_unstable();
    values
}

/// A `Scalar` of `ptype` holding `value`, which must be in range for it.
fn scalar_of(ptype: PType, value: u64) -> Scalar {
    crate::array::scalar_from_bits(&DType::Primitive(ptype, NonNullable), value)
        .vortex_expect("value fits the ptype")
}

fn scalars(array: &ArrayRef) -> VortexResult<Vec<Scalar>> {
    let mut ctx = SESSION.create_execution_ctx();
    (0..array.len())
        .map(|i| array.execute_scalar(i, &mut ctx))
        .collect()
}

// ── Roundtrip over the shapes that change the layout ────────────────────

#[rstest]
// Single element, and the smallest sequences at all.
#[case::one(1, 0)]
#[case::one_sparse(1, 1 << 40)]
#[case::two(2, 1)]
// A dense run: the universe is no larger than the element count, so there are no low bits at all.
#[case::dense(1000, 999)]
// All values equal: one high-part bucket, `lower_width == 0`, and n duplicates.
#[case::all_equal(500, 0)]
// The ordinary sparse case, and one sparse enough to want many low bits.
#[case::sparse(1000, 1 << 20)]
#[case::very_sparse(1000, 1 << 50)]
// Around the FastLanes block boundary, where the low-bits child gains a partial block.
#[case::block_low(1023, 1 << 20)]
#[case::block_exact(1024, 1 << 20)]
#[case::block_high(1025, 1 << 20)]
#[case::two_blocks_low(2047, 1 << 20)]
#[case::two_blocks_exact(2048, 1 << 20)]
#[case::two_blocks_high(2049, 1 << 20)]
// Long enough that both sample tables are non-empty: one-samples need n > 256, and zero-samples
// need more than 512 unset bits, which follows from the upper array being about 2n bits.
#[case::sampled(5000, 1 << 30)]
#[case::sampled_dense(5000, 6000)]
fn test_roundtrip(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let values = sorted_values(n, span, 0x5EED_0001 ^ n as u64);
    let expected = PrimitiveArray::from_iter(values.iter().copied());
    let encoded = encode(&values)?;

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(encoded.len(), n);
    assert_arrays_eq!(encoded, expected, &mut ctx);

    let expected_scalars = scalars(&expected.into_array())?;
    check_access(&encoded, &expected_scalars, 0xC0FFEE)?;
    check_searches(
        &encoded,
        &expected_scalars,
        &probes(&values, span, encoded.dtype(), 0xBEEF),
    )?;
    Ok(())
}

/// Both sample tables must actually be populated at the sizes the roundtrip cases use, or those
/// cases would be silently testing only the unsampled path.
#[test]
fn test_sample_tables_are_exercised() -> VortexResult<()> {
    let encoded = encode(&sorted_values(5000, 1 << 30, 0xDEAD))?;
    let samples = encoded.samples_buffer().len() / size_of::<u64>();
    let num_samples0 = encoded.num_samples0() as usize;
    assert_eq!(num_samples0, 15, "zero-samples");
    assert_eq!(samples - num_samples0, 19, "one-samples");
    Ok(())
}

/// [`params::encoded_bit_size`] is the compressor's cost model, so it has to agree with the
/// encoder exactly rather than approximately. The shapes below are the layout-relevant ones: no low
/// bits, a partial FastLanes block, an exact block, and both sample tables populated.
#[rstest]
#[case::one(1, 0)]
#[case::dense(1000, 999)]
#[case::all_equal(500, 0)]
#[case::sparse(1000, 1 << 20)]
#[case::block_low(1023, 1 << 20)]
#[case::block_exact(1024, 1 << 20)]
#[case::block_high(1025, 1 << 20)]
#[case::one_sample_exact(256, 1 << 16)]
#[case::one_sample_over(257, 1 << 16)]
#[case::sampled(5000, 1 << 30)]
#[case::sampled_dense(5000, 6000)]
fn test_encoded_bit_size_matches_encoder(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let values = sorted_values(n, span, 0x5126_0001 ^ n as u64);
    let encoded = encode(&values)?;

    // The generator draws within `span`, so the sequence's own span is what the encoder saw.
    let actual_span = values[n - 1] - values[0];
    let estimated = params::encoded_bit_size(actual_span, n)?;

    assert_eq!(
        estimated,
        encoded.into_array().nbytes() * 8,
        "n {n}, span {actual_span}"
    );
    Ok(())
}

#[test]
fn test_encoded_bit_size_of_empty() -> VortexResult<()> {
    let encoded = encode::<u64>(&[])?;
    assert_eq!(
        params::encoded_bit_size(0, 0)?,
        encoded.into_array().nbytes() * 8
    );
    Ok(())
}

/// The bulk decode and the per-element cursor must agree — they share no code beyond the layout.
#[rstest]
#[case(1, 0)]
#[case(300, 1 << 16)]
#[case(5000, 1 << 30)]
fn test_decode_matches_cursor(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let values = sorted_values(n, span, 0xABCD);
    let encoded = encode(&values)?;
    let mut ctx = SESSION.create_execution_ctx();

    let decoded = encoded
        .clone()
        .into_array()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    for (index, &want) in decoded.as_slice::<u64>().iter().enumerate() {
        let element = cursor.access_element(index)?;
        assert_eq!(element + values[0], want, "index {index}");
    }
    Ok(())
}

// ── Slicing ────────────────────────────────────────────────────────────

/// A slice records a rank offset and keeps the buffers whole, so every read has to apply it. The
/// starts below straddle the one-sample spacing (256) and the FastLanes block size (1024).
#[rstest]
#[case(0, 1)]
#[case(0, 3000)]
#[case(1, 2999)]
#[case(255, 300)]
#[case(256, 300)]
#[case(257, 300)]
#[case(1023, 1200)]
#[case(1024, 1200)]
#[case(1025, 1200)]
#[case(2999, 3000)]
fn test_slice(#[case] start: usize, #[case] end: usize) -> VortexResult<()> {
    let values = sorted_values(3000, 1 << 24, 0xF00D);
    let encoded = encode(&values)?;
    let sliced = encoded.slice(start..end)?;

    // The slice must stay Elias-Fano rather than falling back to a generic `SliceArray`.
    assert!(
        sliced.is::<EliasFano>(),
        "slice reduced away from EliasFano"
    );
    let sliced = sliced.as_::<EliasFano>().into_owned();
    assert_eq!(sliced.first_rank(), start as u64);
    // The low-bits child is deliberately *not* sliced: one rank offset serves both halves.
    assert_eq!(sliced.lower().len(), values.len());

    let expected = PrimitiveArray::from_iter(values[start..end].iter().copied());
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(sliced, expected, &mut ctx);

    let expected_scalars = scalars(&expected.into_array())?;
    check_access(&sliced, &expected_scalars, 0x1234)?;
    check_searches(
        &sliced,
        &expected_scalars,
        &probes(&values, 1 << 24, sliced.dtype(), 0x5678),
    )?;
    Ok(())
}

/// Slicing twice must compose, and the second slice must not re-slice the child.
#[test]
fn test_slice_of_slice() -> VortexResult<()> {
    let values = sorted_values(2000, 1 << 20, 0x9999);
    let encoded = encode(&values)?;
    let sliced = encoded.slice(500..1500)?.slice(200..800)?;
    assert_eq!(sliced.as_::<EliasFano>().first_rank(), 700);

    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        sliced,
        PrimitiveArray::from_iter(values[700..1300].iter().copied()),
        &mut ctx
    );
    Ok(())
}

// ── Element types ──────────────────────────────────────────────────────

/// Every integer ptype, signed and unsigned, including references at the bottom of the range where
/// the element domain wraps through the whole width.
#[test]
fn test_signed_and_unsigned() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    macro_rules! check {
        ($values:expr) => {{
            let values = $values;
            let expected = PrimitiveArray::from_iter(values.iter().copied());
            let encoded = elias_fano_encode(expected.as_ref().as_::<Primitive>(), &mut ctx)?;
            assert_arrays_eq!(encoded, expected, &mut ctx);
            let expected_scalars = scalars(&expected.into_array())?;
            check_access(&encoded, &expected_scalars, 0x2468)?;
        }};
    }

    check!([0u8, 1, 7, 200, 255]);
    check!([i8::MIN, -100, 0, 100, i8::MAX]);
    check!([0u16, 300, 65535]);
    check!([i16::MIN, 0, i16::MAX]);
    check!([0u32, 1 << 20, u32::MAX]);
    check!([i32::MIN, -1, 0, 1, i32::MAX]);
    check!([0u64, 1 << 40, u64::MAX]);
    check!([i64::MIN, -1, 0, 1, i64::MAX]);
    // Single elements at the extremes, which is where `lower_width` clamps.
    check!([u64::MAX]);
    check!([i64::MIN]);
    Ok(())
}

/// `next_geq` and `rank` over signed columns, probed *inside* the universe.
///
/// [`probes`] builds its set in the unsigned element domain, so every case that calls
/// [`check_next_geq`] drives it with a `u64` column; the signed cases above check `access` alone,
/// and [`test_probes_outside_the_universe`] only reaches the universe edges. That leaves the probe
/// classification in `locate` — which tells below from above in the ptype's own ordering, after a
/// subtraction that wraps for every value under the reference — covered for signed columns only
/// indirectly, through the two bounds `compare` asks for.
#[test]
fn test_signed_next_geq() -> VortexResult<()> {
    macro_rules! check {
        ($ptype:expr, $values:expr) => {{
            let values = $values;
            let encoded = encode(&values)?;
            let expected: Vec<Scalar> = values
                .iter()
                .map(|&v| scalar_of($ptype, v as i64 as u64))
                .collect();
            // Every element, and both its neighbours, so the probe lands on a value, between two,
            // and inside a run of duplicates.
            let mut probes: Vec<Scalar> = values
                .iter()
                .flat_map(|&v| [v.saturating_sub(1), v, v.saturating_add(1)])
                .map(|v| scalar_of($ptype, v as i64 as u64))
                .collect();
            check_searches(&encoded, &expected, &probes)?;
            // And again descending, which reseats backwards rather than walking forwards.
            probes.reverse();
            check_searches(&encoded, &expected, &probes)?;
        }};
    }

    check!(PType::I8, [i8::MIN, -100, -7, -7, 0, 1, 100, i8::MAX]);
    check!(PType::I16, [i16::MIN, -3000, -7, -7, 0, 9, i16::MAX]);
    check!(PType::I32, [i32::MIN, -70_000, -7, -7, 0, 5000, i32::MAX]);
    check!(
        PType::I64,
        [i64::MIN, -1 << 40, -7, -7, 0, 1 << 40, i64::MAX]
    );
    // A reference above zero, so the element domain does not wrap and the negative probes below it
    // all classify as `Bound::Below`.
    check!(PType::I32, [5i32, 5, 900, 1_000_000, i32::MAX]);
    // And one entirely below zero, where every value is a negative pattern.
    check!(PType::I32, [i32::MIN, i32::MIN + 1, -900_000, -5]);
    Ok(())
}

/// `lower_width` on either side of every native width, where a naive implementation would try to
/// bit-pack at or above the child's own width.
#[rstest]
#[case(7)]
#[case(8)]
#[case(9)]
#[case(15)]
#[case(16)]
#[case(17)]
#[case(31)]
#[case(32)]
#[case(33)]
#[case(62)]
#[case(63)]
fn test_lower_width_boundaries(#[case] width: u8) -> VortexResult<()> {
    // `lower_width` is `floor(log2(universe / n))`, so `n` elements over a universe of `n << width`
    // land on exactly `width`. The cap keeps that universe inside 64 bits for the widest cases.
    let n = 400usize.min(1usize << (64 - u32::from(width)).min(20));
    let span = u64::try_from(((n as u128) << width) - 1)?;
    let mut values = sorted_values(n, span, 0x7777 + u64::from(width));
    // Pin the extremes, so the *observed* span is the one the case asked for.
    values[0] = 0;
    values[n - 1] = span;
    let encoded = encode(&values)?;
    assert_eq!(encoded.lower_width(), width);

    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        encoded,
        PrimitiveArray::from_iter(values.iter().copied()),
        &mut ctx
    );
    check_access(
        &encoded,
        &values
            .iter()
            .map(|&v| scalar_of(PType::U64, v))
            .collect::<Vec<_>>(),
        0x8888,
    )?;
    Ok(())
}

// ── Degenerate inputs ──────────────────────────────────────────────────

#[test]
fn test_empty() -> VortexResult<()> {
    let encoded = encode::<u64>(&[])?;
    assert_eq!(encoded.len(), 0);

    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(encoded, PrimitiveArray::empty::<u64>(NonNullable), &mut ctx);
    Ok(())
}

/// A probe below every element must report rank 0 and the first element, and one above every
/// element must report rank `len` and nothing. Each probe below gets a freshly opened cursor, which
/// is not yet seated on anything, so the classification has to come from the universe bounds alone.
#[test]
fn test_probes_outside_the_universe() -> VortexResult<()> {
    let values: Vec<i32> = vec![-5, 0, 10, 10, 400];
    let encoded = encode(&values)?;
    let mut ctx = SESSION.create_execution_ctx();

    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    assert_eq!(cursor.rank(&scalar_of(PType::I32, 401))?, 5);
    assert_eq!(cursor.next_geq(&scalar_of(PType::I32, 401))?.1, None);

    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    let (rank, value) = cursor.next_geq(&scalar_of(PType::I32, i32::MIN as i64 as u64))?;
    assert_eq!(rank, 0);
    assert_eq!(value, Some(scalar_of(PType::I32, -5i32 as i64 as u64)));

    // The maximum is inside the universe, so it must be found rather than clamped away.
    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    assert_eq!(cursor.rank(&scalar_of(PType::I32, 400))?, 4);
    Ok(())
}

/// `rank_inclusive` counts the elements *at or below* the probe, which is the rank of the probe's
/// successor — and at the top of the universe the probe has none.
///
/// [`check_searches`] drives this from every roundtrip and slice case, but its probes all sit
/// inside a universe narrower than the ptype, so the overflow guard and the two
/// outside-the-universe arms need naming here. A span filling the whole width is what makes the
/// successor overflow.
#[test]
fn test_rank_inclusive_at_the_universe_edges() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    // Duplicates at the top, so the count is not simply the length.
    let encoded = encode(&[0u64, 1 << 40, u64::MAX, u64::MAX])?;
    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    assert_eq!(cursor.rank_inclusive(&scalar_of(PType::U64, u64::MAX))?, 4);
    assert_eq!(
        cursor.rank_inclusive(&scalar_of(PType::U64, u64::MAX - 1))?,
        2
    );
    assert_eq!(cursor.rank_inclusive(&scalar_of(PType::U64, 0))?, 1);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, u64::MAX))?, 2);

    // The same at the top of a signed ptype, where the maximum's bit pattern is not the largest
    // `u64` but the span is still the full width.
    let encoded = encode(&[i64::MIN, -1, i64::MAX])?;
    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    assert_eq!(
        cursor.rank_inclusive(&scalar_of(PType::I64, i64::MAX as u64))?,
        3
    );
    assert_eq!(
        cursor.rank_inclusive(&scalar_of(PType::I64, i64::MIN as u64))?,
        1
    );

    // Below the reference and above the maximum, where the count comes from the universe bounds
    // alone rather than from a search.
    let encoded = encode(&[10i32, 20, 20, 30])?;
    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    assert_eq!(cursor.rank_inclusive(&scalar_of(PType::I32, 9))?, 0);
    assert_eq!(cursor.rank_inclusive(&scalar_of(PType::I32, 31))?, 4);
    assert_eq!(cursor.rank_inclusive(&scalar_of(PType::I32, 20))?, 3);

    let empty = encode::<u64>(&[])?;
    let mut cursor = EliasFanoCursor::try_new(empty.as_view(), &mut ctx)?;
    assert_eq!(cursor.rank_inclusive(&scalar_of(PType::U64, 7))?, 0);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 7))?, 0);
    Ok(())
}

/// Duplicates are legal, and a rank must count from the *first* occurrence. This is the case a
/// cursor that trusts wherever it happens to be seated gets wrong.
#[test]
fn test_duplicates_report_first_occurrence() -> VortexResult<()> {
    let values: Vec<u64> = vec![5, 5, 5, 5, 5, 9, 9, 12];
    let encoded = encode(&values)?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;

    // Walk past the duplicates first, so the cursor is seated in the middle of the run...
    assert_eq!(cursor.access(3)?, scalar_of(PType::U64, 5));
    // ...then ask for their value, which must still answer 0.
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 5))?, 0);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 9))?, 5);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 12))?, 7);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 6))?, 5);
    Ok(())
}

/// Many duplicates spread over a wide universe, so runs of equal values share a bucket while the
/// buckets themselves are sparse.
#[test]
fn test_long_duplicate_runs() -> VortexResult<()> {
    let mut values: Vec<u64> = Vec::new();
    for group in 0..40u64 {
        for _ in 0..50 {
            values.push(group * 1000);
        }
    }
    let encoded = encode(&values)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        encoded,
        PrimitiveArray::from_iter(values.iter().copied()),
        &mut ctx
    );

    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;
    for group in 0..40u64 {
        assert_eq!(
            cursor.rank(&scalar_of(PType::U64, group * 1000))?,
            group as usize * 50,
            "group {group}"
        );
    }
    Ok(())
}

/// Probes that advance by a bucket or two while a long run of duplicates sits between them.
///
/// This is the shape that separates the two bounds on the forward walk. The gap in buckets is
/// small enough to walk, but each bucket holds a thousand elements, so the walk has to give up on
/// its element budget and reseat instead of stepping through the run. Every answer here is also
/// reachable by a linear scan, which is what makes the reseat observable only as work not done.
#[test]
fn test_ascending_probes_across_deep_buckets() -> VortexResult<()> {
    // Three runs, placed so the first two share adjacent buckets and the third is far away. The
    // span keeps `lower_width` at 8, so a bucket spans 256 values.
    let mut values: Vec<u64> = Vec::new();
    for &value in &[0u64, 300, 900_000] {
        values.extend(std::iter::repeat_n(value, 1000));
    }
    let encoded = encode(&values)?;
    assert_eq!(encoded.lower_width(), 8, "case depends on the bucket width");

    let mut ctx = SESSION.create_execution_ctx();
    let mut cursor = EliasFanoCursor::try_new(encoded.as_view(), &mut ctx)?;

    // Seat the cursor at the head of the first run, then probe forward. Every probe below sits
    // within `LINEAR_SCAN_THRESHOLD` buckets of the seat but a thousand elements past it.
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 0))?, 0);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 1))?, 1000);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 300))?, 1000);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 301))?, 2000);
    assert_eq!(cursor.rank(&scalar_of(PType::U64, 900_000))?, 2000);

    let decoded = PrimitiveArray::from_iter(values.iter().copied()).into_array();
    let expected_scalars = scalars(&decoded)?;
    check_searches(
        &encoded,
        &expected_scalars,
        &probes(&values, 900_000, encoded.dtype(), 0x7070),
    )
}

/// Clustered data, where buckets are deep, checked against a linear scan over hundreds of probes.
///
/// The sliced half matters as much as the whole: a search inside a bucket has to clamp the bucket
/// to the slice at both ends, and a bucket running past the slice is the case that reports "nothing
/// at or above this" rather than an element the slice does not contain.
#[rstest]
#[case::few_deep(3, 1000, 900_000)]
#[case::many_shallow(200, 25, 1 << 30)]
#[case::mixed(20, 200, 1 << 24)]
fn test_clustered_next_geq(
    #[case] clusters: usize,
    #[case] per_cluster: usize,
    #[case] span: u64,
) -> VortexResult<()> {
    let seed = 0x9E37 ^ span;
    let values = clustered_values(clusters, per_cluster, span, seed);
    let encoded = encode(&values)?.into_array();

    let mut arrays = vec![encoded.clone()];
    arrays.push(encoded.slice(per_cluster / 2..values.len() - per_cluster / 2)?);

    for array in &arrays {
        let array = array.as_::<EliasFano>().into_owned();
        let expected = scalars(&array.clone().into_array())?;
        check_searches(
            &array,
            &expected,
            &probes(&values, span, array.dtype(), seed ^ 0xFEED),
        )?;
    }
    Ok(())
}

#[test]
fn test_rejects_unsorted_and_nullable() {
    let mut ctx = SESSION.create_execution_ctx();
    assert!(encode(&[3u64, 1, 2]).is_err());
    // Nulls have no position in an ordering, so they are refused rather than worked around.
    let nullable = PrimitiveArray::from_option_iter([Some(1u64), None, Some(3)]);
    assert!(elias_fano_encode(nullable.as_ref().as_::<Primitive>(), &mut ctx).is_err());
}

// ── The low-bits child in shapes a rewrite or a file roundtrip can produce ──

/// The child does not have to be a bare `BitPacked`. A file roundtrip can hand it back wrapped, and
/// a rewrite can replace it outright, so both the bulk decode and the cursor must fall back rather
/// than downcast blindly.
#[test]
fn test_unpacked_lower_child() -> VortexResult<()> {
    let values = sorted_values(2000, 1 << 20, 0x4321);
    let encoded = encode(&values)?;
    let mut ctx = SESSION.create_execution_ctx();

    // Replace the bit-packed child with the plain primitive array it decodes to.
    let plain = encoded
        .lower()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
    let rebuilt = rebuild_with_lower(&encoded, plain)?;

    assert_arrays_eq!(
        rebuilt,
        PrimitiveArray::from_iter(values.iter().copied()),
        &mut ctx
    );
    let expected_scalars = values
        .iter()
        .map(|&v| scalar_of(PType::U64, v))
        .collect::<Vec<_>>();
    check_access(&rebuilt, &expected_scalars, 0x1111)?;
    check_searches(
        &rebuilt,
        &expected_scalars,
        &probes(&values, 1 << 20, rebuilt.dtype(), 0x1112),
    )?;

    // And sliced. A cursor over this slot materialises it, so it materialises only the slice's own
    // range and rebases into it — the one place two bases have to be reconciled. The seek paths
    // matter more here than the point lookups: a mishandled base comes back as a wrong rank.
    const START: usize = 1500;
    let sliced = rebuilt.into_array().slice(START..2000)?;
    assert!(
        sliced.is::<EliasFano>(),
        "slice reduced away from EliasFano"
    );
    assert_arrays_eq!(
        sliced,
        PrimitiveArray::from_iter(values[START..2000].iter().copied()),
        &mut ctx
    );
    let sliced = sliced.as_::<EliasFano>().into_owned();
    let sliced_scalars = values[START..2000]
        .iter()
        .map(|&v| scalar_of(PType::U64, v))
        .collect::<Vec<_>>();
    check_access(&sliced, &sliced_scalars, 0x2222)?;
    check_searches(
        &sliced,
        &sliced_scalars,
        &probes(&values, 1 << 20, sliced.dtype(), 0x2223),
    )?;
    Ok(())
}

/// A child carrying a non-zero FastLanes sub-block offset. `unpack_single_primitive` does not apply
/// that offset itself, so a reader that forgets it returns wrong values with no panic anywhere.
#[test]
fn test_lower_child_with_block_offset() -> VortexResult<()> {
    let values = sorted_values(2000, 1 << 20, 0x2222);
    let encoded = encode(&values)?;
    let width = encoded.lower_width();

    // Rebuild the low bits with `pad` junk values in front, then slice them back off. The child now
    // holds the same n values at the same ranks, but starting part-way into a block.
    const PAD: usize = 5;
    let mut padded: Vec<u64> = vec![0; PAD];
    let reference = values[0];
    padded.extend(
        values
            .iter()
            .map(|&v| (v - reference) & params::lower_mask(width)),
    );
    let packed = unsafe {
        bitpack_encode_unchecked(
            PrimitiveArray::new(
                padded.into_iter().collect::<Buffer<u64>>(),
                Validity::NonNullable,
            ),
            width,
        )
    }?
    .into_array()
    .slice(PAD..PAD + values.len())?;
    assert_eq!(packed.as_::<vortex_fastlanes::BitPacked>().offset(), 5);

    let rebuilt = rebuild_with_lower(&encoded, packed)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        rebuilt,
        PrimitiveArray::from_iter(values.iter().copied()),
        &mut ctx
    );
    let expected_scalars = values
        .iter()
        .map(|&v| scalar_of(PType::U64, v))
        .collect::<Vec<_>>();
    check_access(&rebuilt, &expected_scalars, 0x3333)?;
    check_searches(
        &rebuilt,
        &expected_scalars,
        &probes(&values, 1 << 20, rebuilt.dtype(), 0x3334),
    )?;
    Ok(())
}

/// The low bits are OR-ed in under `lower_width`, so a child packed above that width would bleed
/// into the high part. Its width is metadata, so this is refused at construction; a child packed
/// *below* it is a legal tightening and must still be accepted.
#[test]
fn test_rejects_overwide_lower_child() -> VortexResult<()> {
    let values = sorted_values(2000, 1 << 20, 0x6060);
    let encoded = encode(&values)?;
    let width = encoded.lower_width();
    assert!(width > 1, "case needs room on both sides of the width");

    // Masking to the packing width keeps every repack lossless, so the only thing that varies
    // between these two children is the width the layout is asked to accept.
    let pack = |bit_width: u8| -> VortexResult<ArrayRef> {
        let mask = params::lower_mask(bit_width);
        let low: Buffer<u64> = values.iter().map(|&v| (v - values[0]) & mask).collect();
        let packed = unsafe {
            bitpack_encode_unchecked(PrimitiveArray::new(low, Validity::NonNullable), bit_width)
        }?;
        Ok(packed.into_array())
    };

    assert!(
        rebuild_with_lower(&encoded, pack(width + 1)?).is_err(),
        "a child packed wider than lower_width must be rejected"
    );
    rebuild_with_lower(&encoded, pack(width - 1)?)?;
    Ok(())
}

/// Bits above `lower_width` in the low-bits child must be masked off, not trusted.
///
/// A bit-packed child's width is metadata, so [`test_rejects_overwide_lower_child`] refuses that at
/// construction. A patched or rewritten slot arrives as a plain `u64` array instead, where nothing
/// bounds the values at all — and a bit that survives into the high part is a wrong answer with no
/// error anywhere. Both readers are covered: the bulk decode, and the cursor, whose bucket
/// bisection compares low parts directly as well as assembling them.
#[test]
fn test_lower_child_with_junk_above_the_width() -> VortexResult<()> {
    let values = sorted_values(2000, 1 << 20, 0x4949);
    let encoded = encode(&values)?;
    let width = encoded.lower_width();
    assert!(width > 0, "case needs low bits to mask");
    let mut ctx = SESSION.create_execution_ctx();

    let junk = !params::lower_mask(width);
    let plain: Buffer<u64> = encoded
        .lower()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .as_slice::<u64>()
        .iter()
        .map(|&low| low | junk)
        .collect();
    let rebuilt = rebuild_with_lower(
        &encoded,
        PrimitiveArray::new(plain, Validity::NonNullable).into_array(),
    )?;

    let expected = PrimitiveArray::from_iter(values.iter().copied());
    assert_arrays_eq!(rebuilt, expected, &mut ctx);
    let expected_scalars = scalars(&expected.into_array())?;
    check_access(&rebuilt, &expected_scalars, 0x4950)?;
    check_searches(
        &rebuilt,
        &expected_scalars,
        &probes(&values, 1 << 20, rebuilt.dtype(), 0x4951),
    )?;
    Ok(())
}

fn rebuild_with_lower(array: &EliasFanoArray, lower: ArrayRef) -> VortexResult<EliasFanoArray> {
    let len = array.len();
    EliasFano::try_new(array.as_view().data().clone(), lower, len)
}

// ── Statistics and conformance ─────────────────────────────────────────

#[rstest]
#[case::empty(0, 0)]
#[case::single(1, 0)]
#[case::single_sparse(1, 1 << 40)]
#[case::pair(2, 1)]
#[case::all_equal(500, 0)]
#[case::dense(1000, 999)]
#[case::sparse(2000, 1 << 40)]
fn test_is_sorted_stat(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let encoded = encode(&sorted_values(n, span, 0xAAAA + n as u64))?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let mut arrays = vec![encoded.clone()];
    if n > 3 {
        arrays.push(encoded.slice(1..n - 1)?);
    }
    for array in arrays {
        // Present without reading a buffer, which is what `ListArray::new` requires of offsets.
        assert_eq!(
            array
                .statistics()
                .with_typed_stats_set(|stats| stats.get_as::<bool>(Stat::IsSorted)),
            Precision::Exact(true),
            "IsSorted over {} elements",
            array.len()
        );
        // Strictness is declined rather than answered, so it must come back from the generic path
        // with the same answer the decoded array gives.
        let decoded = array
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        assert_eq!(
            array.statistics().compute_is_strict_sorted(&mut ctx),
            decoded.statistics().compute_is_strict_sorted(&mut ctx),
            "IsStrictSorted over {} elements",
            array.len()
        );
    }
    Ok(())
}

#[rstest]
#[case::empty(0, 0)]
#[case::single(1, 0)]
#[case::single_sparse(1, 1 << 40)]
#[case::pair(2, 1)]
#[case::all_equal(500, 0)]
#[case::dense(1000, 999)]
#[case::sparse(2000, 1 << 40)]
fn test_min_max_kernel(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let values = sorted_values(n, span, 0xBBBB + n as u64);
    let encoded = encode(&values)?.into_array();

    // On a slice, the metadata bounds are the *encoded* universe rather than the slice's own
    // extremes, so the slices are what catch a kernel reading metadata instead of elements. The
    // one-element slices are also the shape `min_max` itself uses internally.
    let mut arrays = vec![(encoded.clone(), values.clone())];
    if n > 3 {
        for (start, end) in [(0, n - 1), (1, n), (1, n - 1), (n / 2, n / 2 + 1)] {
            arrays.push((encoded.slice(start..end)?, values[start..end].to_vec()));
        }
    }

    let mut ctx = SESSION.create_execution_ctx();
    for (array, expected) in arrays {
        // The oracle is the decoded array, so the kernel is checked against the generic path and
        // not only against the input it was built from.
        let decoded = array
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        let len = array.len();
        for (name, got, oracle, want) in [
            (
                "min",
                array.statistics().compute_min::<u64>(&mut ctx),
                decoded.statistics().compute_min::<u64>(&mut ctx),
                expected.first().copied(),
            ),
            (
                "max",
                array.statistics().compute_max::<u64>(&mut ctx),
                decoded.statistics().compute_max::<u64>(&mut ctx),
                expected.last().copied(),
            ),
        ] {
            assert_eq!(got, want, "{name} over {len} elements");
            assert_eq!(
                got, oracle,
                "{name} disagrees with the generic path over {len}"
            );
        }
    }
    Ok(())
}

/// `min_max` on a signed column, where the reference is negative and the element domain wraps
/// through the whole width on the way out.
#[test]
fn test_min_max_kernel_signed() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in [
        vec![i32::MIN, -7, -7, 0, i32::MAX],
        vec![-9i32],
        vec![i32::MIN, i32::MAX],
    ] {
        let encoded = encode(&values)?.into_array();
        let decoded = encoded
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        assert_eq!(
            encoded.statistics().compute_min::<i32>(&mut ctx),
            Some(values[0]),
            "min of {values:?}"
        );
        assert_eq!(
            encoded.statistics().compute_max::<i32>(&mut ctx),
            values.last().copied(),
            "max of {values:?}"
        );
        assert_eq!(
            encoded.statistics().compute_min::<i32>(&mut ctx),
            decoded.statistics().compute_min::<i32>(&mut ctx),
            "min disagrees with the generic path for {values:?}"
        );
        assert_eq!(
            encoded.statistics().compute_max::<i32>(&mut ctx),
            decoded.statistics().compute_max::<i32>(&mut ctx),
            "max disagrees with the generic path for {values:?}"
        );
    }
    Ok(())
}

/// The shared compute harnesses, over both a whole array and a slice of it.
///
/// The slice is the half that matters: take and filter both decode arbitrary index sets, which is
/// where a mishandled `first_rank` — the one number a slice records — shows up.
#[rstest]
#[case(1, 0)]
#[case(5, 100)]
#[case(1000, 999)]
#[case(2000, 1 << 20)]
#[case(5000, 1 << 40)]
fn test_conformance(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let encoded = encode(&sorted_values(n, span, 0xCCCC + n as u64))?.into_array();
    let mut arrays = vec![encoded.clone()];
    if n > 2 {
        arrays.push(encoded.slice(1..n - 1)?);
    }

    let mut ctx = SESSION.create_execution_ctx();
    for array in &arrays {
        test_array_consistency(array, &mut ctx);
        test_take_conformance(array, &mut ctx);
        test_filter_conformance(array, &mut ctx);
        test_cast_conformance(array, &mut ctx);
    }
    Ok(())
}

// ── Take and filter pushdown ────────────────────────────────────────────

/// Take and filter along the *cursor* path, not the bulk decode.
///
/// Both kernels hand a dense request back to the framework to decode in one pass, so only a sparse
/// one reaches the per-element cursor — the half that has to apply `first_rank` itself. A
/// conformance harness that happens to ask for many rows would exercise the bulk path and leave
/// this untested, which is why the index sets here are deliberately tiny.
#[rstest]
#[case(2000, 1 << 20)]
#[case(5000, 1 << 40)]
fn test_sparse_take_and_filter(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let values = sorted_values(n, span, 0x5A5A + n as u64);
    let encoded = encode(&values)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    for array in [encoded.clone(), encoded.slice(7..n - 7)?] {
        let decoded = array
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();

        // Sixteen rows spread across thousands: far under `BULK_DECODE_THRESHOLD`, so every read
        // goes through the cursor.
        let picks: Vec<usize> = (0..16).map(|i| i * array.len() / 16).collect();

        // Take sees them in descending order, so a cursor that only ever steps forward fails here.
        let descending = PrimitiveArray::from_iter(picks.iter().rev().map(|&i| i as u64));
        let indices = descending.into_array();
        assert_arrays_eq!(
            array.take(indices.clone())?,
            decoded.take(indices)?,
            &mut ctx
        );

        let mask = Mask::from_indices(array.len(), picks.iter().copied());
        assert_arrays_eq!(array.filter(mask.clone())?, decoded.filter(mask)?, &mut ctx);
    }
    Ok(())
}

/// A null index selects nothing, so its payload is not a position: it is whatever the slot happened
/// to hold, and the cursor path must neither bounds-check nor read it.
///
/// `test_take_conformance` cannot catch this. It builds its nullable indices with
/// `from_option_iter`, which writes a zero under every null — in bounds for any non-empty array, so
/// a reader that ignores validity still passes. The payloads below are the ones that do not: a
/// negative, and two far past the end.
#[test]
fn test_take_ignores_null_index_payloads() -> VortexResult<()> {
    let values = sorted_values(2000, 1 << 20, 0x7A7A);
    let encoded = encode(&values)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    // Six rows out of two thousand: well under `BULK_DECODE_THRESHOLD`, so this is the cursor path
    // rather than the bulk decode, which hands the whole job to the framework.
    let raw: Buffer<i64> = [7i64, i64::MAX, 1500, -1, 99_999, 3].into_iter().collect();
    let indices = PrimitiveArray::new(
        raw,
        Validity::from_mask(Mask::from_indices(6, [0, 2, 5]), Nullable),
    )
    .into_array();

    assert_arrays_eq!(
        encoded.take(indices)?,
        PrimitiveArray::from_option_iter([
            Some(values[7]),
            None,
            Some(values[1500]),
            None,
            None,
            Some(values[3]),
        ]),
        &mut ctx
    );

    // All-null indices are answered before a cursor is opened, which would otherwise materialise a
    // low-bits child for a read that never happens. The constant is the visible sign of that.
    let all_null = PrimitiveArray::new(
        [i64::MIN, -3, 88_888].into_iter().collect::<Buffer<i64>>(),
        Validity::AllInvalid,
    );
    let taken = encoded
        .take(all_null.into_array())?
        .execute::<ArrayRef>(&mut ctx)?;
    assert_eq!(taken.len(), 3);
    let constant = taken
        .as_constant()
        .vortex_expect("an all-null take must reduce to a constant");
    assert!(constant.is_null());
    Ok(())
}

// ── Comparison pushdown ─────────────────────────────────────────────────

/// Every comparison operator, against the decoded array's own answer.
///
/// The pushdown resolves each one to a range through two sampled searches instead of decoding, so
/// it shares no code with the generic path — which is what makes that path a real oracle. Probes
/// cover a value that is present, one that falls between two elements, one below the minimum and
/// one above the maximum, and both ends of the array.
#[rstest]
#[case(1, 0)]
#[case(5, 100)]
#[case(1000, 999)]
#[case(2000, 1 << 20)]
#[case(5000, 1 << 40)]
fn test_compare_matches_decoded(#[case] n: usize, #[case] span: u64) -> VortexResult<()> {
    let values = sorted_values(n, span, 0xC0DE + n as u64);
    let encoded = encode(&values)?.into_array();
    let mut arrays = vec![encoded.clone()];
    if n > 2 {
        arrays.push(encoded.slice(1..n - 1)?);
    }

    let mut probes = vec![0u64, 1, u64::MAX, span, span.saturating_add(1)];
    probes.push(values[0]);
    probes.push(values[n - 1]);
    probes.push(values[n / 2]);
    probes.push(values[n / 2].saturating_add(1));
    probes.push(values[n / 2].saturating_sub(1));

    let mut ctx = SESSION.create_execution_ctx();
    for array in &arrays {
        // The oracle: the same array, decoded, compared by the generic path.
        let decoded = array
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        for &probe in &probes {
            // A nullable literal must not change which rows match, but it does make the result
            // nullable — which a comparison of set bits alone would not notice.
            for literal in nullabilities(Scalar::from(probe))? {
                for op in COMPARISONS {
                    let expr = op(root(), lit(literal.clone()));
                    let pushed = array.clone().apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;
                    let expected = decoded
                        .clone()
                        .apply(&expr)?
                        .execute::<ArrayRef>(&mut ctx)?;
                    assert_eq!(
                        pushed.dtype(),
                        expected.dtype(),
                        "dtype for {literal} over {} elements",
                        array.len()
                    );
                    assert_arrays_eq!(pushed, expected, &mut ctx);
                }
            }
        }
    }
    Ok(())
}

/// `scalar` as itself and again with a nullable dtype, which the comparison kernels have to carry
/// into the result even though the value is never null.
fn nullabilities(scalar: Scalar) -> VortexResult<[Scalar; 2]> {
    let nullable = scalar.cast(&scalar.dtype().as_nullable())?;
    Ok([scalar, nullable])
}

/// The six comparison operators, as expression constructors.
const COMPARISONS: [fn(Expression, Expression) -> Expression; 6] =
    [eq, not_eq, lt, lt_eq, gt, gt_eq];

/// A probe at the very top of a universe that fills a `u64`.
///
/// Every operator but `Lt` and `Gte` needs the count of elements *at or below* the probe, which is
/// the rank of its successor — and here the probe has no successor. Taking it in the value domain,
/// or without a guard in the element domain, overflows: a panic in debug and a wrong answer in
/// release, on the one input that reaches it.
#[test]
fn test_compare_at_the_top_of_the_universe() -> VortexResult<()> {
    let values = [0u64, 1 << 40, u64::MAX];
    let encoded = encode(&values)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let decoded = encoded
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();

    for probe in [u64::MAX, u64::MAX - 1, 0] {
        for op in COMPARISONS {
            let expr = op(root(), lit(Scalar::from(probe)));
            let pushed = encoded
                .clone()
                .apply(&expr)?
                .execute::<ArrayRef>(&mut ctx)?;
            let expected = decoded
                .clone()
                .apply(&expr)?
                .execute::<ArrayRef>(&mut ctx)?;
            assert_eq!(pushed.dtype(), expected.dtype(), "dtype for probe {probe}");
            assert_arrays_eq!(pushed, expected, &mut ctx);
        }
    }
    Ok(())
}

/// A signed, narrow column, probed on and off its elements and at both ends of its range.
///
/// The u64 cases above never exercise the narrowing on the way out, nor a reference below zero,
/// where the element domain wraps through the whole width.
#[test]
fn test_compare_signed_narrow_column() -> VortexResult<()> {
    let values = [i32::MIN, -7, -7, 0, 1, 90, i32::MAX];
    let encoded = encode(&values)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let decoded = encoded
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();

    for probe in [
        i32::MIN,
        i32::MIN + 1,
        -8,
        -7,
        -6,
        0,
        89,
        i32::MAX - 1,
        i32::MAX,
    ] {
        for literal in nullabilities(Scalar::from(probe))? {
            for op in COMPARISONS {
                let expr = op(root(), lit(literal.clone()));
                let pushed = encoded
                    .clone()
                    .apply(&expr)?
                    .execute::<ArrayRef>(&mut ctx)?;
                let expected = decoded
                    .clone()
                    .apply(&expr)?
                    .execute::<ArrayRef>(&mut ctx)?;
                assert_eq!(pushed.dtype(), expected.dtype(), "dtype for {literal}");
                assert_arrays_eq!(pushed, expected, &mut ctx);
            }
        }
    }
    Ok(())
}

/// A comparison every row satisfies must come back as a constant, not a buffer of set bits.
///
/// This is the one externally visible sign that the kernel ran at all: the generic path decodes and
/// compares element by element, so it can only ever produce a `BoolArray`. Without it an
/// unregistered kernel would leave every other test in this file passing.
#[test]
fn test_compare_is_pushed_down() -> VortexResult<()> {
    let values = sorted_values(1000, 1 << 20, 0xF00D);
    let encoded = encode(&values)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    // Every element is >= the minimum, and none is < it.
    let minimum = lit(Scalar::from(values[0]));
    for (expr, expected) in [
        (gt_eq(root(), minimum.clone()), true),
        (lt(root(), minimum), false),
    ] {
        let result = encoded
            .clone()
            .apply(&expr)?
            .execute::<ArrayRef>(&mut ctx)?;
        let constant = result
            .as_constant()
            .vortex_expect("an all-or-nothing comparison must reduce to a constant");
        assert_eq!(constant, Scalar::from(expected));
    }
    Ok(())
}

// ── Serialization, and use as a list column's offsets ───────────────────

/// Serialize, decode, and read back. This is the path a file roundtrip takes, and the only one that
/// exercises `deserialize` — including that it can size the low-bits child, which after a slice is
/// not the array's own length.
#[rstest]
#[case(0, 3000)]
#[case(700, 2100)]
fn test_serde_roundtrip(#[case] start: usize, #[case] end: usize) -> VortexResult<()> {
    let values = sorted_values(3000, 1 << 24, 0xE11A);
    let array = encode(&values)?.into_array().slice(start..end)?;
    let dtype = array.dtype().clone();
    let len = array.len();

    let array_ctx = ArrayContext::empty();
    let mut concat = ByteBufferMut::empty();
    for buffer in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        concat.extend_from_slice(buffer.as_ref());
    }
    let decoded = SerializedArray::try_from(concat.freeze())?.decode(
        &dtype,
        len,
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;

    assert!(decoded.is::<EliasFano>(), "decoded away from EliasFano");
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        decoded,
        PrimitiveArray::from_iter(values[start..end].iter().copied()),
        &mut ctx
    );
    check_access(
        &decoded.as_::<EliasFano>().into_owned(),
        &values[start..end]
            .iter()
            .map(|&v| scalar_of(PType::U64, v))
            .collect::<Vec<_>>(),
        0x9A9A,
    )?;
    Ok(())
}

/// One column that a sorted integer encoding lands under: a list's offsets.
///
/// Worth its own test not because it is what the encoding is for, but because it drives the array
/// through a parent that reads it a boundary at a time. `ListArray::new` refuses offsets that do
/// not report `IsSorted`, and `offset_at` only fast-paths a `Primitive` child — so every list
/// boundary here goes through `scalar_at`, and through the cursor underneath it.
#[test]
fn test_list_offsets() -> VortexResult<()> {
    // Lengths including several empty lists, which is what makes duplicate offsets ordinary.
    let lengths: Vec<u64> = (0..500u64).map(|i| (i * 7) % 5).collect();
    let mut offsets: Vec<u64> = Vec::with_capacity(lengths.len() + 1);
    offsets.push(0);
    for length in &lengths {
        offsets.push(offsets[offsets.len() - 1] + length);
    }
    let total = *offsets.last().vortex_expect("at least one offset") as usize;

    let elements = PrimitiveArray::from_iter((0..total as i32).map(|i| i * 3)).into_array();
    let list = ListArray::try_new(
        elements.clone(),
        encode(&offsets)?.into_array(),
        Validity::NonNullable,
    )?;
    assert_eq!(list.len(), lengths.len());

    let mut ctx = SESSION.create_execution_ctx();
    for (index, &length) in lengths.iter().enumerate() {
        let slice = list.list_elements_at(index)?;
        assert_eq!(slice.len(), length as usize, "list {index} length");
        assert_arrays_eq!(
            slice,
            elements.slice(offsets[index] as usize..offsets[index + 1] as usize)?,
            &mut ctx
        );
    }
    Ok(())
}

// ── Corrupt arrays must raise, never panic ─────────────────────────────

/// A sample table is fed straight to `BitBuffer::select_range` as a window start, and that asserts
/// on a start past the end. So a file with the right sample *count* and garbage sample *values* has
/// to be rejected at construction, not left to panic on the first query.
#[rstest]
#[case::past_the_end(u64::MAX)]
#[case::just_past_the_end(u64::MAX - 1)]
#[case::out_of_order(0)]
fn test_rejects_corrupt_samples(#[case] poison: u64) -> VortexResult<()> {
    // Long enough that both sample tables are populated, so either can be poisoned.
    let encoded = encode(&sorted_values(5000, 1 << 30, 0xDEFACED))?;
    let samples = encoded.samples_buffer();
    assert!(samples.len() >= 2 * size_of::<u64>(), "need two samples");

    for index in [0usize, samples.len() / size_of::<u64>() - 1] {
        let mut poisoned = samples.clone().into_mut();
        let start = index * size_of::<u64>();
        poisoned[start..start + size_of::<u64>()].copy_from_slice(&poison.to_le_bytes());

        let data = EliasFanoData::try_new(
            encoded.upper_buffer().clone(),
            poisoned.freeze(),
            encoded.reference_scalar().clone(),
            encoded.max_scalar().clone(),
            encoded.lower_width(),
            encoded.upper_len(),
            encoded.first_rank(),
        )?;
        let rebuilt = EliasFano::try_new(data, encoded.lower().clone(), encoded.len());
        assert!(
            rebuilt.is_err(),
            "a sample of {poison} at index {index} must be rejected"
        );
    }
    Ok(())
}

/// The upper array's *contents* are not validated — that would mean walking the whole buffer on
/// every construction — so both the bulk decode and the cursor have to raise on a malformed one
/// rather than underflow or hand back a short answer.
#[test]
fn test_corrupt_upper_array_raises() -> VortexResult<()> {
    let encoded = encode(&sorted_values(600, 1 << 16, 0xBADB175))?;
    let mut ctx = SESSION.create_execution_ctx();

    // Set the sentinel at bit 0. Now the first set bit sits at its own rank, so recovering its high
    // part would underflow.
    let upper = encoded.upper_buffer();
    let mut poisoned = upper.clone().into_mut();
    poisoned[0] |= 1;

    let data = EliasFanoData::try_new(
        poisoned.freeze(),
        encoded.samples_buffer().clone(),
        encoded.reference_scalar().clone(),
        encoded.max_scalar().clone(),
        encoded.lower_width(),
        encoded.upper_len(),
        encoded.first_rank(),
    )?;
    let rebuilt = EliasFano::try_new(data, encoded.lower().clone(), encoded.len())?;

    // All three entry points must return an error. Each recovers a high part by subtracting a rank
    // from a bit position, which this input drives negative, so an unchecked subtraction would
    // panic in debug and hand back wrong values in release.
    assert!(
        rebuilt
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err(),
        "bulk decode of a malformed upper array must raise"
    );
    let mut cursor = EliasFanoCursor::try_new(rebuilt.as_view(), &mut ctx)?;
    assert!(
        cursor.access(0).is_err(),
        "cursor access into a malformed upper array must raise"
    );
    assert!(
        rebuilt.execute_scalar(0, &mut ctx).is_err(),
        "scalar_at into a malformed upper array must raise"
    );
    Ok(())
}
