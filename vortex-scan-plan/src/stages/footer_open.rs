// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Footer open: the only stage in this slice whose IO goes through the protocol.

use std::sync::Arc;

use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::DeserializeStep;
use vortex_file::EOF_SIZE;
use vortex_file::Footer;
use vortex_file::FooterDeserializer;
use vortex_file::MAX_POSTSCRIPT_SIZE;
use vortex_io::VortexReadAt;
use vortex_session::VortexSession;

use crate::io::IoConsumer;
use crate::io::IoRequest;
use crate::io::IoRequestId;
use crate::io::IoResult;
use crate::io::IoTarget;
use crate::next::Next;
use crate::planner::Planner;
use crate::planner::PlannerOutput;
use crate::planner::State;
use crate::planner::WorkScope;
use crate::stages::FileSource;
use crate::stages::OpenedFile;

/// The opener's default tail read: enough to hold the largest postscript and the EOF marker.
pub const DEFAULT_INITIAL_READ_SIZE: usize = MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE;

/// Opens a file: discovers its size if unknown, reads and parses the footer unless one was
/// supplied, validates the footer against the size, and hands an [`OpenedFile`] to `next`.
///
/// Request ids increase from zero per instance and stay stable across repeated `state()` calls
/// until delivered. A delivery with an unknown id or the wrong result variant is reported from
/// the next `compute()`.
pub struct FooterOpen {
    read: Arc<dyn VortexReadAt>,
    initial_read_size: usize,
    session: VortexSession,
    next: Next<OpenedFile>,
    phase: Phase,
    next_id: u32,
    error: Option<VortexError>,
}

/// One outstanding range request and, once delivered, its bytes.
struct RangeRead {
    id: IoRequestId,
    offset: u64,
    len: usize,
    received: Option<BufferHandle>,
}

impl RangeRead {
    fn request(&self) -> IoRequest {
        IoRequest {
            request: self.id,
            target: IoTarget::Range {
                offset: self.offset,
                len: self.len,
            },
        }
    }
}

enum Phase {
    /// Footer and size are both known; the next compute emits the child.
    Cached { size: u64, footer: Footer },
    /// Waiting for the source length.
    NeedSize {
        id: IoRequestId,
        footer: Option<Footer>,
    },
    /// Waiting for, or holding, the initial tail read.
    NeedTail { size: u64, tail: RangeRead },
    /// Parsing, possibly waiting for one more range the deserializer asked for.
    Deserialising {
        size: u64,
        deserializer: FooterDeserializer,
        pending: Option<RangeRead>,
    },
    /// The child was emitted; nothing remains.
    Emitted,
}

impl FooterOpen {
    /// Creates the stage. `initial_read_size` is raised to the opener's minimum tail and
    /// clamped to the file size.
    pub fn new(
        source: FileSource,
        initial_read_size: usize,
        session: VortexSession,
        next: Next<OpenedFile>,
    ) -> Self {
        let mut stage = Self {
            read: source.read,
            initial_read_size,
            session,
            next,
            phase: Phase::Emitted,
            next_id: 0,
            error: None,
        };
        stage.phase = match (source.footer, source.size) {
            (Some(footer), Some(size)) => Phase::Cached { size, footer },
            (footer, None) => Phase::NeedSize {
                id: stage.fresh_id(),
                footer,
            },
            (None, Some(size)) => stage.tail_phase(size),
        };
        stage
    }

    fn fresh_id(&mut self) -> IoRequestId {
        let id = IoRequestId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Mirrors `VortexOpenOptions::read_footer`: at least the minimum tail, at most the file.
    fn tail_phase(&mut self, size: u64) -> Phase {
        let len = self.initial_read_size.max(DEFAULT_INITIAL_READ_SIZE);
        let len = usize::try_from(size).map_or(len, |size| len.min(size));
        Phase::NeedTail {
            size,
            tail: RangeRead {
                id: self.fresh_id(),
                offset: size - len as u64,
                len,
                received: None,
            },
        }
    }

    fn emit(&mut self, size: u64, footer: Footer) -> VortexResult<PlannerOutput> {
        footer.validate_file_size(size)?;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..footer.row_count(),
        };
        let child = (self.next)(OpenedFile {
            read: Arc::clone(&self.read),
            size,
            footer,
        })?;
        self.phase = Phase::Emitted;
        Ok(PlannerOutput::Planner(scope, child))
    }

    fn deserialize(&mut self) -> VortexResult<PlannerOutput> {
        let Phase::Deserialising {
            size,
            deserializer,
            pending,
        } = &mut self.phase
        else {
            vortex_bail!("FooterOpen: deserialize called outside the deserialising phase");
        };
        if let Some(read) = pending.take() {
            let Some(handle) = read.received else {
                vortex_bail!(
                    "FooterOpen: compute called while request {:?} is outstanding",
                    read.id
                );
            };
            deserializer.prefix_data(handle.try_into_host_sync()?);
        }
        match deserializer.deserialize()? {
            DeserializeStep::NeedMoreData { offset, len } => {
                let id = IoRequestId(self.next_id);
                self.next_id += 1;
                let read = RangeRead {
                    id,
                    offset,
                    len,
                    received: None,
                };
                let request = read.request();
                let Phase::Deserialising { pending, .. } = &mut self.phase else {
                    vortex_bail!("FooterOpen: phase changed during deserialisation");
                };
                *pending = Some(read);
                Ok(PlannerOutput::NeedsIO(vec![request]))
            }
            DeserializeStep::NeedFileSize => {
                vortex_bail!(
                    "FooterOpen: deserializer asked for the file size after it was supplied"
                )
            }
            DeserializeStep::Done(footer) => {
                let size = *size;
                self.emit(size, footer)
            }
        }
    }
}

impl IoConsumer for FooterOpen {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        let outcome = match (&mut self.phase, result) {
            (Phase::NeedSize { id, footer }, IoResult::Size(size)) if *id == request => {
                let footer = footer.take();
                self.phase = match footer {
                    Some(footer) => Phase::Cached { size, footer },
                    None => self.tail_phase(size),
                };
                Ok(())
            }
            (Phase::NeedTail { tail, .. }, IoResult::Bytes(handle))
                if tail.id == request && tail.received.is_none() =>
            {
                tail.received = Some(handle);
                Ok(())
            }
            (
                Phase::Deserialising {
                    pending: Some(read),
                    ..
                },
                IoResult::Bytes(handle),
            ) if read.id == request && read.received.is_none() => {
                read.received = Some(handle);
                Ok(())
            }
            (_, result) => Err(vortex_err!(
                "FooterOpen: unexpected delivery of {request:?} ({})",
                match result {
                    IoResult::Size(_) => "size",
                    IoResult::Bytes(_) => "bytes",
                }
            )),
        };
        if let Err(error) = outcome {
            self.error.get_or_insert(error);
        }
    }
}

impl Planner for FooterOpen {
    fn state(&self) -> State {
        if self.error.is_some() {
            return State::NeedsCompute;
        }
        match &self.phase {
            Phase::Cached { .. } => State::NeedsCompute,
            Phase::NeedSize { id, .. } => State::NeedsIO(vec![IoRequest {
                request: *id,
                target: IoTarget::Size,
            }]),
            Phase::NeedTail { tail, .. } => match tail.received {
                None => State::NeedsIO(vec![tail.request()]),
                Some(_) => State::NeedsCompute,
            },
            Phase::Deserialising { pending, .. } => match pending {
                Some(read) if read.received.is_none() => State::NeedsIO(vec![read.request()]),
                _ => State::NeedsCompute,
            },
            Phase::Emitted => State::Done,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        match std::mem::replace(&mut self.phase, Phase::Emitted) {
            Phase::Cached { size, footer } => self.emit(size, footer),
            Phase::NeedTail {
                size,
                tail:
                    RangeRead {
                        received: Some(handle),
                        ..
                    },
            } => {
                let tail = handle.try_into_host_sync()?;
                self.phase = Phase::Deserialising {
                    size,
                    deserializer: Footer::deserializer(tail, self.session.clone()).with_size(size),
                    pending: None,
                };
                self.deserialize()
            }
            phase @ Phase::Deserialising { .. } => {
                self.phase = phase;
                self.deserialize()
            }
            phase @ (Phase::NeedSize { .. } | Phase::NeedTail { .. }) => {
                self.phase = phase;
                vortex_bail!("FooterOpen: compute called while IO is outstanding")
            }
            Phase::Emitted => vortex_bail!("FooterOpen: compute called after Done"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_io::runtime::current::CurrentThreadRuntime;

    use super::*;
    use crate::driver::Driver;
    use crate::io::IoSource;
    use crate::io::ReadAtIoSource;
    use crate::next::next_fn;
    use crate::next::pending;
    use crate::tests::fixtures::DonePlanner;
    use crate::tests::fixtures::LARGE_FOOTER_CHUNKS;
    use crate::tests::fixtures::PanickingReadAt;
    use crate::tests::fixtures::RUNTIME;
    use crate::tests::fixtures::RecordingReadAt;
    use crate::tests::fixtures::SESSION;
    use crate::tests::fixtures::open_buffer;
    use crate::tests::fixtures::recording_child;
    use crate::tests::fixtures::write_large_footer_file;
    use crate::tests::fixtures::write_test_file;

    fn small_file() -> VortexResult<ByteBuffer> {
        write_test_file(&[("numbers", buffer![1u32, 2, 3, 4].into_array())])
    }

    fn io_source(read: &Arc<dyn VortexReadAt>) -> ReadAtIoSource<CurrentThreadRuntime> {
        ReadAtIoSource::new(Arc::clone(read), Arc::new(RUNTIME.clone()))
    }

    /// Delivers every request the stage currently reports through a read-at source.
    fn serve(stage: &mut FooterOpen, read: &Arc<dyn VortexReadAt>) -> VortexResult<State> {
        let state = stage.state();
        if let State::NeedsIO(batch) = &state {
            let source = io_source(read);
            for request in batch {
                stage.set_io_result(request.request, source.perform(&request.target)?);
            }
        }
        Ok(state)
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
            match serve(stage, read)? {
                State::Done => vortex_bail!("stage finished without emitting"),
                State::NeedsIO(_) => continue,
                State::NeedsCompute => match stage.compute()? {
                    output @ PlannerOutput::Planner(..) => return Ok(output),
                    PlannerOutput::NeedsIO(_) | PlannerOutput::Continue => continue,
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
        let State::NeedsIO(batch) = stage.state() else {
            vortex_bail!("expected a tail request");
        };
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].target, IoTarget::Range { offset: 0, len });
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
        let State::NeedsIO(batch) = stage.state() else {
            vortex_bail!("expected a size request");
        };
        assert_eq!(
            batch,
            vec![IoRequest {
                request: IoRequestId(0),
                target: IoTarget::Size
            }]
        );
        serve(&mut stage, &read)?;
        let State::NeedsIO(batch) = stage.state() else {
            vortex_bail!("expected a tail request");
        };
        assert_eq!(
            batch,
            vec![IoRequest {
                request: IoRequestId(1),
                target: IoTarget::Range { offset: 0, len }
            }]
        );
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
    fn ids_are_stable_until_delivered(#[case] known_size: bool) -> VortexResult<()> {
        let buffer = small_file()?;
        let size = known_size.then_some(buffer.len() as u64);
        let (next, _) = recording_child();
        let stage = open(Arc::new(buffer), size, None, next);
        let first = stage.state();
        assert!(matches!(first, State::NeedsIO(_)));
        assert_eq!(stage.state(), first);
        assert_eq!(stage.state(), first);
        Ok(())
    }

    #[rstest]
    #[case::wrong_id(IoRequestId(7), IoResult::Size(1))]
    #[case::wrong_variant(
        IoRequestId(0),
        IoResult::Bytes(BufferHandle::new_host(ByteBuffer::empty()))
    )]
    fn bad_delivery_surfaces_from_compute(
        #[case] id: IoRequestId,
        #[case] result: IoResult,
    ) -> VortexResult<()> {
        let (next, _) = recording_child();
        let mut stage = open(Arc::new(small_file()?), None, None, next);
        stage.set_io_result(id, result);
        assert_eq!(stage.state(), State::NeedsCompute);
        let err = stage.compute().err().map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains("unexpected delivery")),
            "{err:?}"
        );
        Ok(())
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
        let PlannerOutput::Planner(_, child) = run_to_child(&mut stage, &read)? else {
            vortex_bail!("expected a child");
        };
        child.start()?;
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
        let session = SESSION.clone();
        let root_read = Arc::clone(&read);
        let root = pending(move || {
            Ok(Box::new(FooterOpen::new(
                FileSource {
                    read: root_read,
                    size: Some(size),
                    footer,
                },
                DEFAULT_INITIAL_READ_SIZE,
                session,
                next,
            )) as Box<dyn Planner>)
        });
        let batches = Driver::new(Arc::new(io_source(&read)))
            .with_step_limit(64)
            .run(root)?;
        assert!(batches.is_empty());
        assert!(invoked.load(Ordering::SeqCst));
        Ok(())
    }
}
