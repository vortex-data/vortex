// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end footer planning tests with diagnostic range output.

pub mod fixtures;
mod range_morsel;

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
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use crate::Footer;
use vortex_io::VortexReadAt;

use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Driver;
use vortex_io::request::ReadAtIoSource;
use crate::planning::FileSource;
use crate::planning::plan_file;
use crate::planning::tests::range_morsel::RangeOnly;
use vortex_scan::planning::next::next_fn;
use crate::planning::tests::fixtures::FailingReadAt;
use crate::planning::tests::fixtures::PanickingReadAt;
use crate::planning::tests::fixtures::RUNTIME;
use crate::planning::tests::fixtures::RecordingReadAt;
use crate::planning::tests::fixtures::SESSION;
use crate::planning::tests::fixtures::open_buffer;
use crate::planning::tests::fixtures::write_test_file;
use crate::planning::tests::fixtures::ctx;

/// Runs `plan_file` through the driver with a read-at IO source.
fn run(
    read: Arc<dyn VortexReadAt>,
    size: Option<u64>,
    footer: Option<Footer>,
    filter: Option<Expression>,
) -> VortexResult<Vec<Batch>> {
    let root = plan_file(
        FileSource {
            read: Arc::clone(&read),
            size,
            footer,
        },
        filter,
        SESSION.clone(),
        next_fn(|opened| Ok(RangeOnly::new(opened))),
    );
    Driver::new(Arc::new(ReadAtIoSource::new(
        read,
        Arc::new(RUNTIME.clone()),
    )))
    .with_step_limit(10_000)
    .run(root)
}

fn numbers_file() -> VortexResult<ByteBuffer> {
    write_test_file(&[("numbers", buffer![1u32, 2, 3, 4, 5, 6, 7, 8].into_array())])
}

fn range_row(start: u64, end: u64) -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        ("start", buffer![start].into_array()),
        ("end", buffer![end].into_array()),
    ])?
    .into_array())
}

#[test]
fn surviving_file_emits_the_file_range() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let size = buffer.len() as u64;
    let batches = run(
        Arc::new(buffer),
        Some(size),
        None,
        Some(gt(col("numbers"), lit(3u32))),
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
        Some(gt(col("numbers"), lit(100u32))),
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
        None,
    )?;
    assert_eq!(batches.len(), 1);
    assert_arrays_eq!(batches[0].array, range_row(0, 8)?, &mut ctx());
    Ok(())
}

#[test]
fn failing_source_surfaces_the_error() -> VortexResult<()> {
    let size = numbers_file()?.len() as u64;
    let err = run(Arc::new(FailingReadAt(size)), None, None, None)
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
    let batches = run(Arc::new(buffer), Some(size), None, None)?;
    assert!(batches.is_empty());
    Ok(())
}

#[rstest]
#[case::known_size(true, 0)]
#[case::unknown_size(false, 1)]
fn known_and_unknown_size_read_only_the_footer(
    #[case] known: bool,
    #[case] size_calls: usize,
) -> VortexResult<()> {
    let buffer = numbers_file()?;
    let len = buffer.len();
    let recording = Arc::new(RecordingReadAt::new(buffer));
    let batches = run(
        Arc::clone(&recording) as Arc<dyn VortexReadAt>,
        known.then_some(len as u64),
        None,
        None,
    )?;
    assert_eq!(batches.len(), 1);
    assert_eq!(recording.size_calls(), size_calls);
    assert_eq!(recording.reads(), vec![(0, len)]);
    Ok(())
}
