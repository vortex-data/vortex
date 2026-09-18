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
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
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
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::EliasFano;
use crate::EliasFanoArray;
use crate::EliasFanoArraySlotsExt;
use crate::EliasFanoData;
use crate::ef;
use crate::elias_fano_encode;

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

/// Every element, read back through `scalar_at`, must match `expected`.
///
/// Probed in a random order and then in sequence. The path is stateless, so the two must give the
/// same answers — which is what the shuffle is there to catch.
fn check_access(array: &EliasFanoArray, expected: &[Scalar], seed: u64) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for index in Rng(seed).permutation(expected.len()) {
        assert_eq!(
            array.execute_scalar(index, &mut ctx)?,
            expected[index],
            "index {index}"
        );
    }
    for (index, want) in expected.iter().enumerate() {
        assert_eq!(
            &array.execute_scalar(index, &mut ctx)?,
            want,
            "sequential index {index}"
        );
    }
    Ok(())
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
/// a rewrite can replace it outright, so both the bulk decode and the per-element read must fall
/// back rather than downcast blindly.
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

    // And sliced, which is the one place two bases have to be reconciled: the slice's own rank
    // offset, and the offset the slot carries. A mishandled base comes back as a wrong element.
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
            .map(|&v| (v - reference) & ef::lower_mask(width)),
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
        let mask = ef::lower_mask(bit_width);
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
/// error anywhere. Both readers are covered: the bulk decode, and the per-element read.
#[test]
fn test_lower_child_with_junk_above_the_width() -> VortexResult<()> {
    let values = sorted_values(2000, 1 << 20, 0x4949);
    let encoded = encode(&values)?;
    let width = encoded.lower_width();
    assert!(width > 0, "case needs low bits to mask");
    let mut ctx = SESSION.create_execution_ctx();

    let junk = !ef::lower_mask(width);
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
    Ok(())
}

fn rebuild_with_lower(array: &EliasFanoArray, lower: ArrayRef) -> VortexResult<EliasFanoArray> {
    let len = array.len();
    EliasFano::try_new(array.as_view().data().clone(), lower, len)
}

// ── Statistics and conformance ─────────────────────────────────────────

// ── Take and filter pushdown ────────────────────────────────────────────

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
/// boundary here goes through `scalar_at`.
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

/// A sample table is fed straight to `select_range` as a window start, and that asserts on a start
/// past the end. So a file with the right sample *count* and garbage sample *values* has to be
/// rejected at construction, not left to panic on the first query.
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
/// every construction — so both the bulk decode and the per-element read have to raise on a
/// malformed one rather than underflow or hand back a short answer.
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

    // Both entry points must return an error. Each recovers a high part by subtracting a rank from
    // a bit position, which this input drives negative, so an unchecked subtraction would panic in
    // debug and hand back wrong values in release.
    assert!(
        rebuilt
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err(),
        "bulk decode of a malformed upper array must raise"
    );
    assert!(
        rebuilt.execute_scalar(0, &mut ctx).is_err(),
        "scalar_at into a malformed upper array must raise"
    );
    Ok(())
}
