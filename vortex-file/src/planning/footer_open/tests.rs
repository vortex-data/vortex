// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::buffer;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequest;
use vortex_io::request::IoSource;
use vortex_io::request::ReadAtIoSource;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_scan::planning::driver::Driver;
use vortex_scan::planning::next::next_fn;

use super::*;
use crate::planning::tests::fixtures::DonePlanner;
use crate::planning::tests::fixtures::LARGE_FOOTER_CHUNKS;
use crate::planning::tests::fixtures::PanickingReadAt;
use crate::planning::tests::fixtures::RUNTIME;
use crate::planning::tests::fixtures::RecordingReadAt;
use crate::planning::tests::fixtures::SESSION;
use crate::planning::tests::fixtures::open_buffer;
use crate::planning::tests::fixtures::recording_child;
use crate::planning::tests::fixtures::write_large_footer_file;
use crate::planning::tests::fixtures::write_test_file;

fn small_file() -> VortexResult<ByteBuffer> {
    write_test_file(&[("numbers", buffer![1u32, 2, 3, 4].into_array())])
}

fn io_source(read: &Arc<dyn VortexReadAt>) -> ReadAtIoSource<CurrentThreadRuntime> {
    ReadAtIoSource::new(Arc::clone(read), Arc::new(RUNTIME.clone()))
}

/// Performs `batch` through a read-at source and delivers the results to the stage.
fn serve(stage: &mut FooterOpen, read: &Arc<dyn VortexReadAt>, batch: IoBatch) -> VortexResult<()> {
    let source = io_source(read);
    let requests = batch.len();
    source.submit(IoOwnerId(0), batch)?;
    for _ in 0..requests {
        let completion = source.wait()?;
        stage.set_io_result(completion.request, completion.result?);
    }
    Ok(())
}

fn open(
    read: Arc<dyn VortexReadAt>,
    size: Option<u64>,
    footer: Option<Footer>,
    next: Next<OpenedFile>,
) -> FooterOpen {
    FooterOpen::new(
        FileSource { read, size, footer },
        DEFAULT_INITIAL_READ_SIZE,
        SESSION.clone(),
        next,
    )
}

/// Drives the stage by hand until it emits, returning the emitted output.
fn run_to_child(
    stage: &mut FooterOpen,
    read: &Arc<dyn VortexReadAt>,
) -> VortexResult<PlannerOutput> {
    for _ in 0..16 {
        match stage.state() {
            State::Done => vortex_bail!("stage finished without emitting"),
            State::Waiting => vortex_bail!("stage waits after every request was delivered"),
            State::NeedsCompute => match stage.compute()? {
                output @ PlannerOutput::Planner(..) => return Ok(output),
                PlannerOutput::NeedsIO(batch) => serve(stage, read, batch)?,
                PlannerOutput::Continue => continue,
                other => vortex_bail!("unexpected output {other:?}"),
            },
        }
    }
    vortex_bail!("stage did not emit within 16 visits")
}

#[test]
fn cached_footer_needs_no_io() -> VortexResult<()> {
    let buffer = small_file()?;
    let footer = open_buffer(&buffer)?.footer().clone();
    let (next, invoked) = recording_child();
    let mut stage = open(
        Arc::new(PanickingReadAt),
        Some(buffer.len() as u64),
        Some(footer),
        next,
    );
    assert_eq!(stage.state(), State::NeedsCompute);
    let output = stage.compute()?;
    assert!(
        matches!(&output, PlannerOutput::Planner(scope, _) if scope.rows == (0..4)),
        "{output:?}"
    );
    assert_eq!(stage.state(), State::Done);
    assert!(invoked.load(Ordering::SeqCst));
    Ok(())
}

#[test]
fn known_size_reads_the_tail_only() -> VortexResult<()> {
    let buffer = small_file()?;
    let len = buffer.len();
    let size = len as u64;
    let recording = Arc::new(RecordingReadAt::new(buffer));
    let read: Arc<dyn VortexReadAt> = Arc::clone(&recording) as Arc<dyn VortexReadAt>;
    let (next, _) = recording_child();
    let mut stage = open(Arc::clone(&read), Some(size), None, next);
    assert_eq!(stage.state(), State::NeedsCompute);
    let PlannerOutput::NeedsIO(batch) = stage.compute()? else {
        vortex_bail!("expected a tail request");
    };
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].target, IoTarget::range(0, len));
    assert_eq!(stage.state(), State::Waiting);
    serve(&mut stage, &read, batch)?;
    run_to_child(&mut stage, &read)?;
    assert_eq!(recording.size_calls(), 0);
    assert_eq!(recording.reads(), vec![(0, len)]);
    Ok(())
}

#[test]
fn unknown_size_asks_for_size_then_tail() -> VortexResult<()> {
    let buffer = small_file()?;
    let len = buffer.len();
    let read: Arc<dyn VortexReadAt> = Arc::new(buffer);
    let (next, _) = recording_child();
    let mut stage = open(Arc::clone(&read), None, None, next);
    let PlannerOutput::NeedsIO(batch) = stage.compute()? else {
        vortex_bail!("expected a size request");
    };
    assert_eq!(
        batch,
        vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(0),
            target: IoTarget::Size
        }]
    );
    assert_eq!(stage.state(), State::Waiting);
    serve(&mut stage, &read, batch)?;
    assert_eq!(stage.state(), State::NeedsCompute);
    let PlannerOutput::NeedsIO(batch) = stage.compute()? else {
        vortex_bail!("expected the tail request to be published");
    };
    assert_eq!(
        batch,
        vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(1),
            target: IoTarget::range(0, len)
        }]
    );
    assert_eq!(stage.state(), State::Waiting);
    Ok(())
}

#[test]
fn large_footer_needs_a_second_range() -> VortexResult<()> {
    let buffer = write_large_footer_file()?;
    let size = buffer.len() as u64;
    let recording = Arc::new(RecordingReadAt::new(buffer));
    let read: Arc<dyn VortexReadAt> = Arc::clone(&recording) as Arc<dyn VortexReadAt>;
    let (next, _) = recording_child();
    let mut stage = open(Arc::clone(&read), Some(size), None, next);
    let output = run_to_child(&mut stage, &read)?;
    assert!(
        matches!(&output, PlannerOutput::Planner(scope, _) if scope.rows == (0..LARGE_FOOTER_CHUNKS)),
        "{output:?}"
    );
    let reads = recording.reads();
    assert_eq!(reads.len(), 2, "{reads:?}");
    assert_eq!(
        reads[0],
        (
            size - DEFAULT_INITIAL_READ_SIZE as u64,
            DEFAULT_INITIAL_READ_SIZE
        )
    );
    assert!(
        reads[1].0 < reads[0].0,
        "second read precedes the tail: {reads:?}"
    );
    Ok(())
}

#[rstest]
#[case::known_size(true)]
#[case::unknown_size(false)]
fn waits_until_its_request_is_delivered(#[case] known_size: bool) -> VortexResult<()> {
    let buffer = small_file()?;
    let size = known_size.then_some(buffer.len() as u64);
    let read: Arc<dyn VortexReadAt> = Arc::new(buffer);
    let (next, _) = recording_child();
    let mut stage = open(Arc::clone(&read), size, None, next);
    assert_eq!(stage.state(), State::NeedsCompute);
    let PlannerOutput::NeedsIO(batch) = stage.compute()? else {
        vortex_bail!("expected a request");
    };
    assert_eq!(stage.state(), State::Waiting);
    assert_eq!(stage.state(), State::Waiting);
    serve(&mut stage, &read, batch)?;
    assert_eq!(stage.state(), State::NeedsCompute);
    Ok(())
}

#[rstest]
#[case::wrong_id(IoRequestId(7), IoResult::Size(1))]
#[case::wrong_variant(
    IoRequestId(0),
    IoResult::Bytes(BufferHandle::new_host(ByteBuffer::empty()))
)]
#[should_panic(expected = "IoSlot")]
fn bad_delivery_is_a_driver_bug(#[case] id: IoRequestId, #[case] result: IoResult) {
    let (next, _) = recording_child();
    let buffer = small_file().expect("fixture file");
    let mut stage = open(Arc::new(buffer), None, None, next);
    drop(stage.compute().expect("the size request"));
    stage.set_io_result(id, result);
}

#[test]
fn child_receives_size_and_footer() -> VortexResult<()> {
    let buffer = small_file()?;
    let size = buffer.len() as u64;
    let read: Arc<dyn VortexReadAt> = Arc::new(buffer);
    let seen = Arc::new(parking_lot::Mutex::new(None));
    let sink = Arc::clone(&seen);
    let next = next_fn(move |opened: OpenedFile| {
        *sink.lock() = Some((opened.size, opened.footer.row_count()));
        Ok(DonePlanner)
    });
    let mut stage = open(Arc::clone(&read), None, None, next);
    run_to_child(&mut stage, &read)?;
    assert_eq!(*seen.lock(), Some((size, 4)));
    Ok(())
}

#[rstest]
#[case::cached(true)]
#[case::uncached(false)]
fn through_the_driver(#[case] cached: bool) -> VortexResult<()> {
    let buffer = small_file()?;
    let size = buffer.len() as u64;
    let footer = cached
        .then(|| open_buffer(&buffer))
        .transpose()?
        .map(|f| f.footer().clone());
    let read: Arc<dyn VortexReadAt> = if cached {
        Arc::new(PanickingReadAt)
    } else {
        Arc::new(buffer)
    };
    let (next, invoked) = recording_child();
    let root = FooterOpen::new(
        FileSource {
            read: Arc::clone(&read),
            size: Some(size),
            footer,
        },
        DEFAULT_INITIAL_READ_SIZE,
        SESSION.clone(),
        next,
    );
    let batches = Driver::new(Arc::new(io_source(&read)))
        .with_step_limit(64)
        .run(Box::new(root))?;
    assert!(batches.is_empty());
    assert!(invoked.load(Ordering::SeqCst));
    Ok(())
}
