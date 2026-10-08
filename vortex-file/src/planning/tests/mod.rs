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
use vortex_io::VortexReadAt;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_io::request::ReadAtIoSource;
use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Driver;
use vortex_scan::planning::next::next_fn;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::Footer;
use crate::planning::FileSource;
use crate::planning::plan_file;
use crate::planning::tests::fixtures::FailingReadAt;
use crate::planning::tests::fixtures::LifoReadAtIoSource;
use crate::planning::tests::fixtures::PanickingReadAt;
use crate::planning::tests::fixtures::RUNTIME;
use crate::planning::tests::fixtures::RecordingReadAt;
use crate::planning::tests::fixtures::SESSION;
use crate::planning::tests::fixtures::ctx;
use crate::planning::tests::fixtures::open_buffer;
use crate::planning::tests::fixtures::write_test_file;
use crate::planning::tests::range_morsel::RangeOnly;

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

/// A root that hands out several files, one per compute, then finishes.
struct Fanout {
    files: Vec<Box<dyn Planner>>,
    ordinal: u64,
}

impl IoConsumer for Fanout {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Fanout {
    fn state(&self) -> State {
        if self.files.is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let file = self.files.remove(0);
        let scope = WorkScope {
            file_ordinal: self.ordinal,
            rows: 0..0,
        };
        self.ordinal += 1;
        Ok(PlannerOutput::Planner(scope, file))
    }
}

/// Two files planned concurrently over one IO source, both with an unknown size.
fn two_files_root(buffer: &ByteBuffer) -> Box<dyn Planner> {
    let plan = || {
        plan_file(
            FileSource {
                read: Arc::new(buffer.clone()),
                size: None,
                footer: None,
            },
            None,
            SESSION.clone(),
            next_fn(|opened| Ok(RangeOnly::new(opened))),
        )
    };
    let files = vec![plan(), plan()];
    Box::new(Fanout { files, ordinal: 0 })
}

fn expect_two_files(batches: Vec<Batch>) -> VortexResult<()> {
    assert_eq!(batches.len(), 2, "one batch per file");
    for batch in batches {
        assert_arrays_eq!(batch.array, range_row(0, 8)?, &mut ctx());
    }
    Ok(())
}

/// The LIFO source completes the newest submission first, so the second file's size and tail both
/// finish while the first file's size request is still waiting.
#[test]
fn two_files_complete_out_of_order() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let source = Arc::new(LifoReadAtIoSource::new(Arc::new(buffer.clone())));
    let batches = Driver::new(Arc::clone(&source) as Arc<dyn IoSource>)
        .with_step_limit(10_000)
        .run(two_files_root(&buffer))?;

    let submitted = source.submitted();
    let completed = source.completed();
    assert_eq!(submitted.len(), 4, "{submitted:?}");
    let (first, second) = (submitted[0], submitted[1]);
    assert_ne!(
        first, second,
        "both files had a size request in flight before either completed"
    );
    assert_eq!(submitted, vec![first, second, second, first]);
    assert_eq!(completed, vec![second, second, first, first]);
    expect_two_files(batches)
}

#[test]
fn two_files_over_the_read_at_source() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let source: Arc<dyn IoSource> = Arc::new(ReadAtIoSource::new(
        Arc::new(buffer.clone()),
        Arc::new(RUNTIME.clone()),
    ));
    let batches = Driver::new(source)
        .with_step_limit(10_000)
        .run(two_files_root(&buffer))?;
    expect_two_files(batches)
}
