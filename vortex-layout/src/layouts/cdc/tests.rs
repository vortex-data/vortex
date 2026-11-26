// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Tests build synthetic data where lossy numeric casts are harmless.
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;

use super::*;
use crate::layouts::cdc::xet::XET_MAX_CHUNK_SIZE;
use crate::layouts::cdc::xet::XET_MIN_CHUNK_SIZE;
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

fn fixed_feed(rows: &[u64]) -> RowFeed {
    let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    RowFeed {
        marker: MarkerFeed::NonNullable,
        values: ValueFeed::Fixed {
            bytes: ByteBuffer::from(bytes),
            width: 8,
        },
    }
}

fn test_options() -> ContentDefinedChunkingOptions {
    // Small sizes keep tests fast: chunks of roughly 1 KiB + 2^10 bytes.
    ContentDefinedChunkingOptions {
        min_chunk_bytes: 1024,
        max_chunk_bytes: 8192,
        boundary_mask_bits: 10,
    }
}

fn cuts_for(rows: &[u64], options: &ContentDefinedChunkingOptions) -> Vec<usize> {
    let mut cutter = RollingCutter::new(options);
    cutter.process_rows(&[fixed_feed(rows)], rows.len())
}

#[test]
fn cuts_respect_min_and_max_sizes() {
    let rows = random_rows(0, 100_000);
    let options = test_options();
    let cuts = cuts_for(&rows, &options);
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
}

#[test]
fn cuts_are_invariant_to_incoming_chunk_splits() {
    let rows = random_rows(1, 50_000);
    let options = test_options();
    let whole = cuts_for(&rows, &options);

    // Feeding the same rows in arbitrary increments must produce identical boundaries.
    let mut cutter = RollingCutter::new(&options);
    let mut split_cuts = Vec::new();
    let mut offset = 0usize;
    for piece in [1usize, 7, 100, 8192, 1000, 40_700] {
        let end = (offset + piece).min(rows.len());
        let cuts = cutter.process_rows(&[fixed_feed(&rows[offset..end])], end - offset);
        split_cuts.extend(cuts.into_iter().map(|cut| cut + offset));
        offset = end;
    }
    assert_eq!(offset, rows.len());
    assert_eq!(whole, split_cuts);
}

#[test]
fn cuts_resynchronize_after_insert() {
    let options = test_options();
    let v1 = random_rows(2, 100_000);
    // Insert 1000 fresh rows at 40%, leaving every other row's content unchanged.
    let insert_at = 40_000;
    let mut v2 = v1.clone();
    let mut inserted = random_rows(3, 1000);
    v2.splice(insert_at..insert_at, inserted.drain(..));

    let cuts1 = cuts_for(&v1, &options);
    let cuts2 = cuts_for(&v2, &options);

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
}

#[test]
fn cuts_resynchronize_for_low_entropy_columns() {
    // Sequential values (ids, near-constant timestamps) have almost no per-byte entropy. Raw
    // GEAR hashing starves on such input and cut positions degrade into fixed strides that
    // never re-align after a row shift; whitening each row through mix64 restores uniform
    // boundary candidates. This is a regression test for that failure mode.
    let options = test_options();
    let v1: Vec<u64> = (0..200_000u64).map(|i| 1_700_000_000 + i * 1000).collect();
    let delete_at = 120_000;
    let mut v2 = v1.clone();
    v2.drain(delete_at..delete_at + 1000);

    let cuts1 = cuts_for(&v1, &options);
    let cuts2 = cuts_for(&v2, &options);

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
}

#[test]
fn strategy_emits_content_defined_blocks() -> VortexResult<()> {
    let rows = random_rows(4, 200_000);
    let expected_cuts = cuts_for(&rows, &test_options());
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
