// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test doubles, fixtures, and integration tests for the prototype.

pub mod fixtures;
pub mod scripted;

use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_file::Footer;
use vortex_io::VortexReadAt;

use crate::driver::Batch;
use crate::driver::Driver;
use crate::io::ReadAtIoSource;
use crate::stages::FileSource;
use crate::stages::ScanQuery;
use crate::stages::plan_file;
use crate::tests::fixtures::FailingReadAt;
use crate::tests::fixtures::PanickingReadAt;
use crate::tests::fixtures::RUNTIME;
use crate::tests::fixtures::RecordingReadAt;
use crate::tests::fixtures::SESSION;
use crate::tests::fixtures::concat;
use crate::tests::fixtures::open_buffer;
use crate::tests::fixtures::write_chunked_test_file;
use crate::tests::fixtures::write_test_file;
use crate::tests::scripted::ctx;

/// Runs `plan_file` through the driver with a read-at IO source.
fn run(
    read: Arc<dyn VortexReadAt>,
    size: Option<u64>,
    footer: Option<Footer>,
    query: ScanQuery,
) -> VortexResult<Vec<Batch>> {
    let runtime = Arc::new(RUNTIME.clone());
    let root = plan_file(
        FileSource {
            read: Arc::clone(&read),
            size,
            footer,
        },
        query,
        SESSION.clone(),
        Arc::clone(&runtime),
    );
    Driver::new(Arc::new(ReadAtIoSource::new(read, runtime)))
        .with_step_limit(10_000)
        .run(root)
}

fn range_only(filter: Option<Expression>) -> ScanQuery {
    ScanQuery {
        filter,
        projection: None,
    }
}

fn numbers_file() -> VortexResult<ByteBuffer> {
    write_test_file(&[("numbers", buffer![1u32, 2, 3, 4, 5, 6, 7, 8].into_array())])
}

fn two_chunk_file() -> VortexResult<(ByteBuffer, Vec<ArrayRef>)> {
    let chunks = vec![
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ];
    Ok((
        write_chunked_test_file(&[("numbers", chunks.clone())])?,
        chunks,
    ))
}

fn range_row(start: u64, end: u64) -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        ("start", buffer![start].into_array()),
        ("end", buffer![end].into_array()),
    ])?
    .into_array())
}

#[test]
fn survivor_without_projection_emits_the_file_range() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let size = buffer.len() as u64;
    let batches = run(
        Arc::new(buffer),
        Some(size),
        None,
        range_only(Some(gt(col("numbers"), lit(3u32)))),
    )?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].scope.rows, 0..8);
    assert_arrays_eq!(batches[0].array, range_row(0, 8)?, &mut ctx());
    Ok(())
}

#[test]
fn pruned_file_reads_only_the_footer() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let len = buffer.len();
    let recording = Arc::new(RecordingReadAt::new(buffer));
    let batches = run(
        Arc::clone(&recording) as Arc<dyn VortexReadAt>,
        Some(len as u64),
        None,
        range_only(Some(gt(col("numbers"), lit(100u32)))),
    )?;
    assert!(batches.is_empty());
    assert_eq!(recording.reads(), vec![(0, len)]);
    assert_eq!(recording.size_calls(), 0);
    Ok(())
}

#[test]
fn cached_footer_needs_no_reads() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let footer = open_buffer(&buffer)?.footer().clone();
    let batches = run(
        Arc::new(PanickingReadAt),
        Some(buffer.len() as u64),
        Some(footer),
        range_only(None),
    )?;
    assert_eq!(batches.len(), 1);
    assert_arrays_eq!(batches[0].array, range_row(0, 8)?, &mut ctx());
    Ok(())
}

#[test]
fn failing_source_surfaces_the_error() -> VortexResult<()> {
    let size = numbers_file()?.len() as u64;
    let err = run(Arc::new(FailingReadAt(size)), None, None, range_only(None))
        .err()
        .map(|e| e.to_string());
    assert!(
        err.as_deref()
            .is_some_and(|m| m.contains(FailingReadAt::MESSAGE)),
        "{err:?}"
    );
    Ok(())
}

#[test]
fn empty_file_emits_nothing() -> VortexResult<()> {
    let buffer = write_test_file(&[("numbers", buffer![0u32; 0].into_array())])?;
    let size = buffer.len() as u64;
    for projection in [None, Some(root())] {
        let batches = run(
            Arc::new(buffer.clone()),
            Some(size),
            None,
            ScanQuery {
                filter: None,
                projection,
            },
        )?;
        assert!(batches.is_empty());
    }
    Ok(())
}

#[test]
fn projection_over_two_chunks_returns_the_written_array() -> VortexResult<()> {
    let (buffer, chunks) = two_chunk_file()?;
    let size = buffer.len() as u64;
    let batches = run(
        Arc::new(buffer),
        Some(size),
        None,
        ScanQuery {
            filter: None,
            projection: Some(root()),
        },
    )?;
    assert_eq!(batches.len(), 2);
    let scopes: Vec<_> = batches
        .iter()
        .map(|batch| batch.scope.rows.clone())
        .collect();
    assert_eq!(scopes, vec![0..4, 4..8]);
    let expected: Vec<ArrayRef> = chunks
        .into_iter()
        .map(|chunk| Ok(StructArray::from_fields(&[("numbers", chunk)])?.into_array()))
        .collect::<VortexResult<_>>()?;
    let dtype = expected[0].dtype().clone();
    let actual = batches.into_iter().map(|batch| batch.array).collect();
    assert_arrays_eq!(
        concat(actual, &dtype)?,
        concat(expected, &dtype)?,
        &mut ctx()
    );
    Ok(())
}

#[rstest]
#[case::second_chunk_only(6u32, vec![4..8])]
#[case::both_chunks(2u32, vec![0..4, 4..8])]
#[case::none(100u32, vec![])]
fn filter_batches_match_non_empty_splits(
    #[case] threshold: u32,
    #[case] expected_scopes: Vec<std::ops::Range<u64>>,
) -> VortexResult<()> {
    let (buffer, _) = two_chunk_file()?;
    let size = buffer.len() as u64;
    let batches = run(
        Arc::new(buffer),
        Some(size),
        None,
        ScanQuery {
            filter: Some(gt(col("numbers"), lit(threshold))),
            projection: Some(root()),
        },
    )?;
    let scopes: Vec<_> = batches
        .iter()
        .map(|batch| batch.scope.rows.clone())
        .collect();
    assert_eq!(scopes, expected_scopes);
    for batch in &batches {
        assert!(!batch.array.is_empty());
    }
    Ok(())
}
