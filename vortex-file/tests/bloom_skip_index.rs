// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end coverage for the zoned Bloom skipping index.
//!
//! Bloom indexes are optional extensions rather than part of the default file layout. This test
//! exercises the complete opt-in lifecycle:
//!
//! 1. register the index with a write session (with editions disabled) and request it for one field;
//! 2. persist one Bloom filter per zone and reopen the file with a fresh registered session;
//! 3. prove that equality predicates prune zones while returning the same rows as a full scan; and
//! 4. reopen the indexed file with an unregistered, allow-unknown session to verify that the index
//!    is ignorable.
//!
//! The input is intentionally hostile to ordinary min/max pruning. Zone `z` contains values whose
//! remainder modulo [`NZONES`] is `z`, so both [`HIT`] and [`MISS`] lie inside every zone's
//! min/max range. `MISS` is then removed from its zone without changing that range. Consequently,
//! pruning either value requires the Bloom filter rather than the built-in range statistics.

#![expect(clippy::expect_used)]

use std::num::NonZeroU32;
use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::eq;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::lit;
use vortex_array::expr::bound::root;
use vortex_array::field_path;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::BloomOptions;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::HashFn;
use vortex_layout::layouts::zoned::skip_index::SkipIndexRef;
use vortex_layout::layouts::zoned::skip_index::SkipIndexSessionExt;
use vortex_layout::layouts::zoned::skip_index::bloom::BloomSkipIndex;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

const ZONE_LEN: usize = 256;
const NZONES: usize = 4;
const HIT: i64 = 502;
const MISS: i64 = 503;

fn bloom() -> SkipIndexRef {
    bloom_with_options(BloomOptions::default())
}

fn bloom_with_options(options: BloomOptions) -> SkipIndexRef {
    SkipIndexRef::new(Arc::new(BloomSkipIndex::new(options)))
}

fn session(register_bloom: bool) -> VortexSession {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);

    if register_bloom {
        session.register_skip_index::<BloomSkipIndex>();
    }

    session
}

fn data() -> ArrayRef {
    data_with_shape(ZONE_LEN, NZONES, Some(MISS))
}

fn data_with_shape(zone_len: usize, nzones: usize, missing: Option<i64>) -> ArrayRef {
    let chunks = (0..nzones)
        .map(|zone| {
            let mut values = (0..zone_len)
                .map(|row| i64::try_from(row * nzones + zone).expect("test value fits i64"))
                .collect::<Vec<_>>();
            if let Some(missing) = missing
                && usize::try_from(missing).expect("missing value is non-negative") % nzones == zone
            {
                // Leave a hole inside every zone's min/max range so a MISS cannot be pruned by the
                // ordinary range stats. The bloom must provide the proof.
                values[usize::try_from(missing).expect("missing value is non-negative") / nzones] =
                    i64::try_from(zone_len * nzones + zone).expect("replacement fits i64");
            }
            StructArray::from_fields(&[("id", PrimitiveArray::from_iter(values).into_array())])
                .expect("valid test struct")
                .into_array()
        })
        .collect::<Vec<_>>();
    ChunkedArray::try_new(
        chunks,
        DType::struct_(
            [("id", DType::Primitive(PType::I64, Nullability::NonNullable))],
            Nullability::NonNullable,
        ),
    )
    .expect("valid chunked test data")
    .into_array()
}

fn filter(value: i64) -> BoundExpression {
    let input_dtype = DType::struct_(
        [("id", DType::Primitive(PType::I64, Nullability::NonNullable))],
        Nullability::NonNullable,
    );

    eq(get_item("id", root(input_dtype)), lit(value))
}

fn strategy(index: &SkipIndexRef, zone_len: usize) -> VortexResult<Arc<dyn LayoutStrategy>> {
    let mut options = ZonedLayoutOptions {
        block_size: NonZeroUsize::new(zone_len).expect("zone length is non-zero"),
        ..Default::default()
    };

    // Adding the aggregate to these field-specific zoned options is the explicit write-side
    // opt-in. Registering the index in the session alone does not change the file layout.
    options = options.with_skip_index(index.clone());

    Ok(WriteStrategyBuilder::default()
        .with_field_zoned_options(field_path!(id), options)
        .build())
}

async fn scan(file: &vortex_file::VortexFile, value: i64) -> VortexResult<ArrayRef> {
    file.scan()?
        .with_filter(filter(value))
        .into_array_stream()?
        .read_all()
        .await
}

async fn write_file(
    session: &VortexSession,
    input: &ArrayRef,
    index: &SkipIndexRef,
    zone_len: usize,
) -> VortexResult<Vec<u8>> {
    let mut bytes = Vec::new();
    session
        .write_options()
        // Bloom filters are not part of any edition.
        .disable_editions()
        .with_strategy(strategy(index, zone_len)?)
        .write(&mut bytes, input.to_array_stream())
        .await?;
    Ok(bytes)
}

#[expect(clippy::tests_outside_test_module)]
#[tokio::test]
async fn bloom_roundtrip_prunes_and_unknown_reader_matches_full_scan() -> VortexResult<()> {
    let index_ref = bloom();
    let write_session = session(true);
    let input = data();
    let bytes = write_file(&write_session, &input, &index_ref, ZONE_LEN).await?;

    // Reconstruct every read-side extension from a fresh session rather than accidentally relying
    // on state retained by the writer.
    let read_session = session(true);
    let file = read_session.open_options().open_buffer(bytes.clone())?;
    let reader = file.layout_reader()?;
    let row_count = file.row_count();

    // HIT is present only in zone 2. Since it falls within every zone's min/max range, the exact
    // one-zone mask proves that the Bloom falsifier participated in pruning.
    let hit_mask = reader
        .pruning_evaluation(
            &(0..row_count),
            &filter(HIT),
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await?;
    assert_eq!(hit_mask.true_count(), ZONE_LEN);
    assert!(hit_mask.iter().take(2 * ZONE_LEN).all(|keep| !keep));
    assert!(
        hit_mask
            .iter()
            .skip(2 * ZONE_LEN)
            .take(ZONE_LEN)
            .all(|keep| keep)
    );
    assert!(hit_mask.iter().skip(3 * ZONE_LEN).all(|keep| !keep));

    // MISS was removed while remaining inside every zone's min/max range. Only the Bloom filters
    // can prove that all four zones are absent.
    let miss_mask = reader
        .pruning_evaluation(
            &(0..row_count),
            &filter(MISS),
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await?;
    assert!(
        miss_mask.all_false(),
        "an absent value should prune every zone"
    );

    // An allow-unknown reader without Bloom registration bypasses the unavailable zone map and
    // scans the data child. This both supplies the reference result and verifies that an optional
    // index does not become a hard read-time dependency.
    let full_scan_session = session(false);
    full_scan_session.allow_unknown();
    let full_scan_file = full_scan_session.open_options().open_buffer(bytes)?;

    let indexed_hit = scan(&file, HIT).await?;
    let full_scan_hit = scan(&full_scan_file, HIT).await?;
    // A Bloom filter may retain extra zones, but it must never change query results.
    assert_arrays_eq!(
        indexed_hit,
        full_scan_hit,
        &mut read_session.create_execution_ctx()
    );
    let expected_hit =
        StructArray::from_fields(&[("id", PrimitiveArray::from_iter([HIT]).into_array())])?
            .into_array();
    assert_arrays_eq!(
        full_scan_hit,
        expected_hit,
        &mut read_session.create_execution_ctx()
    );

    let indexed_miss = scan(&file, MISS).await?;
    let full_scan_miss = scan(&full_scan_file, MISS).await?;
    assert_arrays_eq!(
        indexed_miss,
        full_scan_miss,
        &mut read_session.create_execution_ctx()
    );
    assert_eq!(full_scan_miss.len(), 0);
    Ok(())
}

#[expect(clippy::tests_outside_test_module)]
#[tokio::test]
async fn reader_uses_bloom_options_serialized_in_file() -> VortexResult<()> {
    let options = BloomOptions::new(
        NonZeroU32::new(512).expect("block count is non-zero"),
        HashFn::XxHash3_64,
    );
    let index_ref = bloom_with_options(options);
    let write_session = session(true);
    let input = data();
    let bytes = write_file(&write_session, &input, &index_ref, ZONE_LEN).await?;

    // The reader registers only the implementation type and does not receive
    // the options used by the writer.
    let read_session = session(true);
    let file = read_session.open_options().open_buffer(bytes)?;
    let reader = file.layout_reader()?;
    let row_count = file.row_count();
    let miss_mask = reader
        .pruning_evaluation(
            &(0..row_count),
            &filter(MISS),
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await?;

    assert!(
        miss_mask.all_false(),
        "an absent value should be pruned using the serialized Bloom options"
    );
    Ok(())
}
