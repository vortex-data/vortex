// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opening a Vortex file from inside a plan.

use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSlot;
use vortex_io::request::IoTarget;
use vortex_io::request::OpenTarget;
use vortex_io::request::SourceId;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use super::compose;
use crate::DeserializeStep;
use crate::EOF_SIZE;
use crate::Footer;
use crate::FooterDeserializer;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VortexFile;

/// Builds the planner that continues an opened file, given the file and the source its requests
/// must name.
pub type ThenFn = Box<dyn FnOnce(VortexFile, SourceId) -> VortexResult<Box<dyn Planner>> + Send>;

/// Opens a Vortex file as part of a plan, then continues into the planner built from it.
///
/// Opening is ordinary planner work: an [`IoTarget::Open`] addressed to the plan's
/// [`SourcesIoService`](vortex_io::request::SourcesIoService), then the footer read through the
/// new source as `NeedsIO`, then the continuation, emitted as a dependent child. Nothing blocks and
/// nothing is registered outside the plan, so a table planner spawns one per kept file and the
/// driver opens them in parallel.
///
/// What the caller knows already saves reads: with [`with_file_size`](Self::with_file_size) the
/// size request is skipped, and with [`with_footer`](Self::with_footer), such as one inlined in a
/// manifest, the file is not read at all before the continuation.
///
/// The opened file's segments are read through the plan's sources, never through a segment
/// source of the file's own, so the continuation must stamp the source it is given on every
/// request. [`OpenFilePlanner::scan`] does that for a V2 scan.
pub struct OpenFilePlanner {
    sources: SourceId,
    target: Option<OpenTarget>,
    session: VortexSession,
    scope: WorkScope,
    file_size: Option<u64>,
    footer: Option<Footer>,
    then: Option<ThenFn>,
    slot: IoSlot,
    phase: Phase,
}

enum Phase {
    /// The open is about to be requested.
    Open,
    /// Waiting for the open.
    Opening,
    /// The file's source, and whatever is still to be read before the footer is known.
    Opened(SourceId),
    /// Waiting for the file's size.
    Sizing(SourceId),
    /// Waiting for the bytes the footer starts with, at the end of the file.
    Tail(SourceId, u64),
    /// Waiting for more bytes the footer asked for.
    More(SourceId, u64, FooterDeserializer),
    /// The footer is known: the continuation is next.
    Ready(SourceId),
    Done,
}

impl OpenFilePlanner {
    /// Opens `target` through the plan's sources service `sources`, then continues with `then`,
    /// emitted over `scope`.
    pub fn new(
        sources: SourceId,
        target: OpenTarget,
        session: VortexSession,
        scope: WorkScope,
        then: ThenFn,
    ) -> Self {
        Self {
            slot: IoSlot::new(sources),
            sources,
            target: Some(target),
            session,
            scope,
            file_size: None,
            footer: None,
            then: Some(then),
            phase: Phase::Open,
        }
    }

    /// Opens `target` and continues into a V2 scan of it, configured by `configure`, composed with
    /// [`compose`]: pruning, then the chunks of the rows it keeps.
    pub fn scan(
        sources: SourceId,
        target: OpenTarget,
        session: VortexSession,
        scope: WorkScope,
        configure: impl FnOnce(ScanBuilder) -> ScanBuilder + Send + 'static,
    ) -> Self {
        let scan_session = session.clone();
        let then: ThenFn = Box::new(move |file: VortexFile, source| {
            let builder = configure(ScanBuilder::new(scan_session, file.scan_file()));
            Ok(compose(&Arc::new(builder.prepare(source)?)))
        });
        Self::new(sources, target, session, scope, then)
    }

    /// The file's size in bytes, known ahead, such as from a listing or a manifest.
    pub fn with_file_size(mut self, file_size: u64) -> Self {
        self.file_size = Some(file_size);
        self
    }

    /// The file's footer, known ahead, such as one inlined in a manifest.
    pub fn with_footer(mut self, footer: Footer) -> Self {
        self.footer = Some(footer);
        self
    }

    fn read(&mut self, source: SourceId, offset: u64, len: usize) -> PlannerOutput {
        PlannerOutput::NeedsIO(self.slot.issue_to(source, IoTarget::range(offset, len)))
    }

    /// Feeds the deserializer the bytes it has, reading more or finishing.
    fn deserialize(
        &mut self,
        source: SourceId,
        size: u64,
        mut deserializer: FooterDeserializer,
    ) -> VortexResult<PlannerOutput> {
        match deserializer.deserialize()? {
            DeserializeStep::NeedMoreData { offset, len } => {
                self.phase = Phase::More(source, size, deserializer);
                Ok(self.read(source, offset, len))
            }
            DeserializeStep::NeedFileSize => vortex_bail!("the file's size was given"),
            DeserializeStep::Done(footer) => {
                footer.validate_file_size(size)?;
                self.footer = Some(footer);
                self.phase = Phase::Ready(source);
                Ok(PlannerOutput::Continue)
            }
        }
    }

    fn take_bytes(&mut self) -> VortexResult<ByteBuffer> {
        match self.slot.take() {
            Some(IoResult::Bytes(bytes)) => bytes.try_to_host_sync(),
            _ => vortex_bail!("OpenFilePlanner: a footer read was not delivered"),
        }
    }
}

impl IoConsumer for OpenFilePlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.slot.deliver(request, result);
    }
}

impl Planner for OpenFilePlanner {
    fn state(&self) -> State {
        match self.phase {
            Phase::Done => State::Done,
            _ if self.slot.is_waiting() => State::Waiting,
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Open => {
                let target = self
                    .target
                    .take()
                    .ok_or_else(|| vortex_err!("OpenFilePlanner opened twice"))?;
                self.phase = Phase::Opening;
                Ok(PlannerOutput::NeedsIO(
                    self.slot.issue_to(self.sources, IoTarget::Open(target)),
                ))
            }
            Phase::Opening => match self.slot.take() {
                Some(IoResult::Opened(source)) => {
                    self.phase = Phase::Opened(source);
                    Ok(PlannerOutput::Continue)
                }
                _ => vortex_bail!("OpenFilePlanner: the open was not delivered"),
            },
            Phase::Opened(source) => {
                if self.footer.is_some() {
                    self.phase = Phase::Ready(source);
                    return Ok(PlannerOutput::Continue);
                }
                match self.file_size {
                    Some(size) => self.read_tail(source, size),
                    None => {
                        self.phase = Phase::Sizing(source);
                        Ok(PlannerOutput::NeedsIO(
                            self.slot.issue_to(source, IoTarget::Size),
                        ))
                    }
                }
            }
            Phase::Sizing(source) => match self.slot.take() {
                Some(IoResult::Size(size)) => self.read_tail(source, size),
                _ => vortex_bail!("OpenFilePlanner: the size was not delivered"),
            },
            Phase::Tail(source, size) => {
                let tail = self.take_bytes()?;
                let deserializer = Footer::deserializer(tail, self.session.clone()).with_size(size);
                self.deserialize(source, size, deserializer)
            }
            Phase::More(source, size, mut deserializer) => {
                deserializer.prefix_data(self.take_bytes()?);
                self.deserialize(source, size, deserializer)
            }
            Phase::Ready(source) => {
                let footer = self
                    .footer
                    .take()
                    .ok_or_else(|| vortex_err!("OpenFilePlanner has no footer"))?;
                let then = self
                    .then
                    .take()
                    .ok_or_else(|| vortex_err!("OpenFilePlanner continued twice"))?;
                let file = VortexFile::new(footer, Arc::new(PlanSegments), self.session.clone());
                Ok(PlannerOutput::Planner(
                    self.scope.clone(),
                    then(file, source)?,
                ))
            }
            Phase::Done => Ok(PlannerOutput::Done),
        }
    }
}

impl OpenFilePlanner {
    fn read_tail(&mut self, source: SourceId, size: u64) -> VortexResult<PlannerOutput> {
        let len = (MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE).min(usize::try_from(size)?);
        self.phase = Phase::Tail(source, size);
        Ok(self.read(source, size - len as u64, len))
    }
}

/// The segment source of a file opened inside a plan, whose segments are read through the plan's
/// sources instead.
struct PlanSegments;

impl SegmentSource for PlanSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        future::ready(Err(vortex_err!(
            "segment {id} of a file opened in a plan is read through the plan's sources"
        )))
        .boxed()
    }
}

#[cfg(test)]
mod tests;
