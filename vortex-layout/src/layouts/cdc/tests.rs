// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Tests build synthetic data where lossy numeric casts are harmless.
#![allow(clippy::cast_possible_truncation)]

use std::ops::Range;
use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VariantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;
use vortex_utils::aliases::hash_set::HashSet;

use super::*;
use crate::layouts::cdc::xet::XET_BOUNDARY_MASK;
use crate::layouts::cdc::xet::XET_MAX_CHUNK_SIZE;
use crate::layouts::cdc::xet::XET_MIN_CHUNK_SIZE;
use crate::layouts::cdc::xet::XET_TARGET_CHUNK_SIZE;
use crate::layouts::cdc::xet::xet_chunks;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::test::new_session;

/// A tiny deterministic PRNG so tests do not depend on the `rand` crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn random_rows(seed: u64, len: usize) -> Vec<u64> {
    let mut rng = SplitMix64(seed);
    (0..len).map(|_| rng.next()).collect()
}

fn digests_of(rows: &[u64]) -> VortexResult<Vec<RowDigest>> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(rows.iter().copied());
    row_digests(&Canonical::Primitive(array), &mut ctx)
}

fn test_options() -> ContentDefinedChunkingOptions {
    // Small sizes keep tests fast: chunks of roughly 1 KiB + 2^10 bytes.
    ContentDefinedChunkingOptions {
        min_chunk_bytes: 1024,
        max_chunk_bytes: 8192,
        boundary_mask_bits: 10,
    }
}

fn cuts_for(rows: &[u64], options: &ContentDefinedChunkingOptions) -> VortexResult<Vec<usize>> {
    Ok(RollingCutter::new(options).process_rows(&digests_of(rows)?))
}

#[test]
fn cuts_respect_min_and_max_sizes() -> VortexResult<()> {
    let rows = random_rows(0, 100_000);
    let options = test_options();
    let cuts = cuts_for(&rows, &options)?;
    assert!(cuts.len() > 10, "expected many cuts, got {}", cuts.len());

    let mut prev = 0usize;
    for &cut in &cuts {
        let chunk_bytes = (cut - prev) * 8;
        assert!(
            chunk_bytes as u64 >= options.min_chunk_bytes,
            "chunk of {chunk_bytes} bytes is below the minimum"
        );
        // A forced cut lands on the first row end at or past the maximum, so a chunk may
        // overshoot by at most one row.
        assert!(
            (chunk_bytes as u64) < options.max_chunk_bytes + 8,
            "chunk of {chunk_bytes} bytes exceeds the maximum"
        );
        prev = cut;
    }
    Ok(())
}

#[test]
fn cuts_are_invariant_to_incoming_chunk_splits() -> VortexResult<()> {
    let rows = random_rows(1, 50_000);
    let options = test_options();
    let whole = cuts_for(&rows, &options)?;

    // Feeding the same rows in arbitrary increments must produce identical boundaries.
    let mut cutter = RollingCutter::new(&options);
    let mut split_cuts = Vec::new();
    let mut offset = 0usize;
    for piece in [1usize, 7, 100, 8192, 1000, 40_700] {
        let end = (offset + piece).min(rows.len());
        let cuts = cutter.process_rows(&digests_of(&rows[offset..end])?);
        split_cuts.extend(cuts.into_iter().map(|cut| cut + offset));
        offset = end;
    }
    assert_eq!(offset, rows.len());
    assert_eq!(whole, split_cuts);
    Ok(())
}

#[test]
fn cuts_resynchronize_after_insert() -> VortexResult<()> {
    let options = test_options();
    let v1 = random_rows(2, 100_000);
    // Insert 1000 fresh rows at 40%, leaving every other row's content unchanged.
    let insert_at = 40_000;
    let mut v2 = v1.clone();
    let mut inserted = random_rows(3, 1000);
    v2.splice(insert_at..insert_at, inserted.drain(..));

    let cuts1 = cuts_for(&v1, &options)?;
    let cuts2 = cuts_for(&v2, &options)?;

    // Cuts strictly before the insertion point must be identical.
    let before1: Vec<_> = cuts1
        .iter()
        .copied()
        .take_while(|c| *c <= insert_at)
        .collect();
    let before2: Vec<_> = cuts2
        .iter()
        .copied()
        .take_while(|c| *c <= insert_at)
        .collect();
    assert_eq!(before1, before2);

    // Cuts must re-align with the content within a few maximum-size chunks after the edit.
    let resync_rows = 3 * (options.max_chunk_bytes as usize / 8);
    let tail1: Vec<_> = cuts1
        .iter()
        .filter(|c| **c > insert_at + resync_rows)
        .map(|c| v1.len() - c)
        .collect();
    let tail2: std::collections::BTreeSet<_> = cuts2.iter().map(|c| v2.len() - c).collect();
    assert!(!tail1.is_empty());
    for end_distance in tail1 {
        assert!(
            tail2.contains(&end_distance),
            "cut at end-distance {end_distance} did not re-synchronize"
        );
    }
    Ok(())
}

#[test]
fn cuts_resynchronize_for_low_entropy_columns() -> VortexResult<()> {
    // Sequential values (ids, near-constant timestamps) have almost no per-byte entropy. Raw
    // GEAR hashing starves on such input and cut positions degrade into fixed strides that
    // never re-align after a row shift; whitening each row through mix64 restores uniform
    // boundary candidates. This is a regression test for that failure mode.
    let options = test_options();
    let v1: Vec<u64> = (0..200_000u64).map(|i| 1_700_000_000 + i * 1000).collect();
    let delete_at = 120_000;
    let mut v2 = v1.clone();
    v2.drain(delete_at..delete_at + 1000);

    let cuts1 = cuts_for(&v1, &options)?;
    let cuts2 = cuts_for(&v2, &options)?;

    // Every cut sufficiently far past the edit must re-align with the content (identical
    // distance from the end of the data).
    let resync_rows = 3 * (options.max_chunk_bytes as usize / 8);
    let tail1: Vec<_> = cuts1
        .iter()
        .filter(|c| **c > delete_at + resync_rows)
        .map(|c| v1.len() - c)
        .collect();
    let tail2: std::collections::BTreeSet<_> = cuts2.iter().map(|c| v2.len() - c).collect();
    assert!(!tail1.is_empty());
    for end_distance in tail1 {
        assert!(
            tail2.contains(&end_distance),
            "cut at end-distance {end_distance} did not re-synchronize"
        );
    }
    Ok(())
}

#[test]
fn strategy_emits_content_defined_blocks() -> VortexResult<()> {
    let rows = random_rows(4, 200_000);
    let expected_cuts = cuts_for(&rows, &test_options())?;
    let array = PrimitiveArray::from_iter(rows.iter().copied());

    let ctx = ArrayContext::empty();
    let segments = Arc::new(TestSegments::default());
    let (ptr, eof) = SequenceId::root().split();

    let child = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
    let strategy = CdcRepartitionStrategy::new(child, test_options());

    let stream = array.into_array().to_array_stream().sequenced(ptr);
    let layout = block_on(|handle| async move {
        let session = new_session().with_handle(handle);
        strategy
            .write_stream(
                ctx.into(),
                Arc::<TestSegments>::clone(&segments),
                stream,
                eof,
                &session,
            )
            .await
    })?;

    assert_eq!(layout.row_count(), 200_000);
    // One child per cut, plus the tail after the last cut (the data is random, so a cut
    // landing exactly at the end has negligible probability).
    assert_eq!(layout.nchildren(), expected_cuts.len() + 1);
    for (i, window) in expected_cuts.windows(2).enumerate() {
        let child = layout
            .slot(i + 1)?
            .ok_or_else(|| vortex_err!("chunk slot missing"))?;
        assert_eq!(child.row_count(), (window[1] - window[0]) as u64);
    }
    Ok(())
}

#[test]
fn xet_chunks_cover_data_within_size_bounds() {
    let mut rng = SplitMix64(5);
    let data: Vec<u8> = (0..1_000_000).map(|_| rng.next() as u8).collect();
    let chunks = xet_chunks(&data);

    let mut expected_start = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.start, expected_start);
        expected_start = chunk.end;
        if i + 1 < chunks.len() {
            assert!(chunk.len() >= XET_MIN_CHUNK_SIZE);
        }
        assert!(chunk.len() <= XET_MAX_CHUNK_SIZE);
    }
    assert_eq!(expected_start, data.len());
}

#[test]
fn xet_chunks_handle_tiny_input() {
    assert!(xet_chunks(&[]).is_empty());
    assert_eq!(xet_chunks(&[1, 2, 3]), vec![0..3]);
}

/// A four-row canonical array of `kind`, whose rows all differ in content.
fn sample_rows(kind: &str) -> VortexResult<Canonical> {
    Ok(match kind {
        "null" => Canonical::Null(NullArray::new(4)),
        "bool" => Canonical::Bool(BoolArray::from_iter([true, false, true, true])),
        "primitive" => Canonical::Primitive(PrimitiveArray::from_iter([1i64, 2, 3, 4])),
        "decimal" => Canonical::Decimal(DecimalArray::from_iter(
            [10i128, 20, 30, 40],
            DecimalDType::new(10, 2),
        )),
        "varbinview" => {
            Canonical::VarBinView(VarBinViewArray::from_iter_str(["a", "bb", "ccc", "dddd"]))
        }
        // Offsets need not ascend and views may overlap, so the sample exercises both.
        "list" => Canonical::List(ListViewArray::new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]).into_array(),
            PrimitiveArray::from_iter([2u32, 0, 3, 1]).into_array(),
            PrimitiveArray::from_iter([2u32, 1, 3, 2]).into_array(),
            Validity::NonNullable,
        )),
        "map" => {
            let map_dtype = MapDType::try_new(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
                false,
            )?;
            let keys = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]).into_array();
            let values =
                VarBinViewArray::from_iter_str(["a", "b", "c", "d", "e", "f"]).into_array();
            let entries = StructArray::try_from_iter([("key", keys), ("value", values)])?;
            let entries = ListViewArray::new(
                entries.into_array(),
                PrimitiveArray::from_iter([0u32, 2, 3, 5]).into_array(),
                PrimitiveArray::from_iter([2u32, 1, 2, 1]).into_array(),
                Validity::NonNullable,
            );
            Canonical::Map(MapArray::try_new(map_dtype, entries)?)
        }
        "fixed_size_list" => Canonical::FixedSizeList(FixedSizeListArray::new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8]).into_array(),
            2,
            Validity::NonNullable,
            4,
        )),
        "struct" => Canonical::Struct(StructArray::try_from_iter([
            ("a", PrimitiveArray::from_iter([1i32, 2, 3, 4]).into_array()),
            (
                "b",
                VarBinViewArray::from_iter_str(["w", "x", "y", "z"]).into_array(),
            ),
        ])?),
        "union" => Canonical::Union(UnionArray::try_new(
            PrimitiveArray::from_iter([5u8, 9, 5, 9]).into_array(),
            UnionVariants::try_new(
                ["number", "flag"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Bool(Nullability::NonNullable),
                ],
                vec![5, 9],
            )?,
            vec![
                PrimitiveArray::from_iter([10i32, 0, 30, 0]).into_array(),
                BoolArray::from_iter([false, true, false, false]).into_array(),
            ],
        )?),
        "extension" => Canonical::Extension(ExtensionArray::new(
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
            PrimitiveArray::from_iter([1i64, 2, 3, 4]).into_array(),
        )),
        "variant" => {
            let core_storage = ChunkedArray::try_new(
                [1i32, 2, 3, 4].map(|value| {
                    ConstantArray::new(
                        Scalar::variant(Scalar::primitive(value, Nullability::NonNullable)),
                        1,
                    )
                    .into_array()
                }),
                DType::Variant(Nullability::NonNullable),
            )?;
            Canonical::Variant(VariantArray::try_new(core_storage.into_array(), None)?)
        }
        _ => vortex_panic!("unknown sample kind {kind}"),
    })
}

/// Every value type must be inspected deeply enough that rows differing in content digest
/// differently. Types whose content the digest pass cannot see would produce one repeated
/// digest, offering the rolling hash no boundary candidates at all.
#[rstest]
fn digests_distinguish_rows_of_every_value_type(
    #[values(
        "null",
        "bool",
        "primitive",
        "decimal",
        "varbinview",
        "list",
        "map",
        "fixed_size_list",
        "struct",
        "union",
        "extension",
        "variant"
    )]
    kind: &str,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = sample_rows(kind)?;
    let digests = row_digests(&array, &mut ctx)?;

    assert_eq!(digests.len(), array.len());
    // Boundaries must be reproducible, or two writes of the same data would not share chunks.
    assert_eq!(digests, row_digests(&array, &mut ctx)?);
    // A row of zero width never advances the chunk budget, so such a column is never cut.
    assert!(
        digests.iter().all(|digest| digest.width > 0),
        "{kind}: rows carry no serialized width"
    );

    let distinct: HashSet<u64> = digests.iter().map(|digest| digest.hash).collect();
    if kind == "null" {
        // A null column genuinely has no content to tell its rows apart.
        assert_eq!(distinct.len(), 1);
    } else {
        assert!(
            distinct.len() > 1,
            "{kind}: rows with differing content digested identically"
        );
    }
    Ok(())
}

/// Nulls must digest from their validity marker alone: the storage behind a null is undefined
/// padding, and letting it reach the digest would tie boundaries to how nulls were encoded.
#[test]
fn null_rows_ignore_the_values_behind_them() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let validity = Validity::from_iter([true, false, true]);
    let mut digests = |values: [i64; 3]| -> VortexResult<Vec<RowDigest>> {
        let array = PrimitiveArray::new(buffer![values[0], values[1], values[2]], validity.clone());
        row_digests(&Canonical::Primitive(array), &mut ctx)
    };

    // The two arrays differ only in the value sitting behind the null in row 1.
    let padded_with_seven = digests([1, 7, 3])?;
    let padded_with_999 = digests([1, 999, 3])?;
    assert_eq!(padded_with_seven, padded_with_999);
    Ok(())
}

/// The same holds one level down: the fields behind a null struct are undefined too.
#[test]
fn null_struct_rows_ignore_their_fields() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let mut digests = |hidden: i32| -> VortexResult<Vec<RowDigest>> {
        let field = PrimitiveArray::from_iter([1i32, hidden, 3]).into_array();
        let array = StructArray::try_new(
            ["a"].into(),
            vec![field],
            3,
            Validity::from_iter([true, false, true]),
        )?;
        row_digests(&Canonical::Struct(array), &mut ctx)
    };

    assert_eq!(digests(7)?, digests(999)?);
    Ok(())
}

#[test]
fn gearhash_default_table_is_the_xet_normative_table() {
    // Cut positions in the write path and the measured Xet chunk boundaries are both functions
    // of this table: an upstream change to it would silently stop new files deduplicating
    // against previously written ones, so pin the table's contents.
    assert_eq!(DEFAULT_TABLE[0], 0xb088_d3a9_e840_f559);
    assert_eq!(DEFAULT_TABLE[255], 0x63c7_a906_c1dd_187b);
    let fnv1a = DEFAULT_TABLE
        .iter()
        .flat_map(|entry| entry.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    assert_eq!(fnv1a, 0xa4c0_4d9d_bc7e_8bbd);
}

/// The Xet chunker exactly as specified: one scalar GEAR update per byte. [`xet_chunks`] must
/// produce identical boundaries with its SIMD scan and minimum-size skipping.
fn xet_reference_chunks(data: &[u8]) -> Vec<Range<usize>> {
    let mut chunks = Vec::new();
    let mut hash = 0u64;
    let mut start = 0usize;
    for (i, &byte) in data.iter().enumerate() {
        hash = (hash << 1).wrapping_add(DEFAULT_TABLE[byte as usize]);
        let size = i + 1 - start;
        if size < XET_MIN_CHUNK_SIZE {
            continue;
        }
        if size >= XET_MAX_CHUNK_SIZE || hash & XET_BOUNDARY_MASK == 0 {
            chunks.push(start..i + 1);
            start = i + 1;
            hash = 0;
        }
    }
    if start < data.len() {
        chunks.push(start..data.len());
    }
    chunks
}

fn xet_test_bytes(pattern: &str, len: usize) -> Vec<u8> {
    let mut rng = SplitMix64(0xC0FFEE);
    (0..len)
        .map(|i| match pattern {
            "random" => rng.next() as u8,
            "zeros" => 0,
            "constant" => 0xAB,
            "cycle" => ((i % 7) * 37) as u8,
            "ramp" => i as u8,
            _ => unreachable!("unknown pattern {pattern}"),
        })
        .collect()
}

#[rstest]
fn xet_chunks_match_the_scalar_reference(
    #[values("random", "zeros", "constant", "cycle", "ramp")] pattern: &str,
    #[values(
        0,
        1,
        63,
        64,
        XET_MIN_CHUNK_SIZE - 1,
        XET_MIN_CHUNK_SIZE,
        XET_MIN_CHUNK_SIZE + 1,
        XET_TARGET_CHUNK_SIZE,
        XET_MAX_CHUNK_SIZE,
        XET_MAX_CHUNK_SIZE + 1,
        300_000,
        1_048_583
    )]
    len: usize,
) {
    let data = xet_test_bytes(pattern, len);
    assert_eq!(xet_chunks(&data), xet_reference_chunks(&data));
}
