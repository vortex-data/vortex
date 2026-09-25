// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Footer open: the only stage in this slice whose IO goes through the protocol.

use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use crate::DeserializeStep;
use crate::EOF_SIZE;
use crate::Footer;
use crate::FooterDeserializer;
use crate::MAX_POSTSCRIPT_SIZE;
use vortex_io::VortexReadAt;
use vortex_session::VortexSession;

use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSlot;
use vortex_io::request::IoTarget;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use crate::planning::FileSource;
use crate::planning::OpenedFile;

/// The opener's default tail read: enough to hold the largest postscript and the EOF marker.
pub const DEFAULT_INITIAL_READ_SIZE: usize = MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE;

/// Opens a file: discovers its size if unknown, reads and parses the footer unless one was
/// supplied, validates the footer against the size, and hands an [`OpenedFile`] to `next`.
///
/// One request is outstanding at a time and its id stays stable across repeated `state()`
/// calls until delivered. A delivery the stage did not ask for is a driver bug and panics.
pub struct FooterOpen {
    read: Arc<dyn VortexReadAt>,
    initial_read_size: usize,
    session: VortexSession,
    next: Next<OpenedFile>,
    phase: Phase,
    io: IoSlot,
}

enum Phase {
    /// Footer and size are both known; the next compute emits the child.
    Cached { size: u64, footer: Footer },
    /// The source length is in flight.
    NeedSize { footer: Option<Footer> },
    /// The initial tail read is in flight.
    NeedTail { size: u64 },
    /// Parsing; a further range the deserializer asked for may be in flight.
    Deserialising {
        size: u64,
        deserializer: FooterDeserializer,
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
            io: IoSlot::default(),
        };
        stage.phase = match (source.footer, source.size) {
            (Some(footer), Some(size)) => Phase::Cached { size, footer },
            (footer, None) => {
                stage.io.issue(IoTarget::Size);
                Phase::NeedSize { footer }
            }
            (None, Some(size)) => {
                stage.issue_tail(size);
                Phase::NeedTail { size }
            }
        };
        stage
    }

    /// Mirrors `VortexOpenOptions::read_footer`: at least the minimum tail, at most the file.
    fn issue_tail(&mut self, size: u64) -> IoBatch {
        let len = self.initial_read_size.max(DEFAULT_INITIAL_READ_SIZE);
        let len = usize::try_from(size).map_or(len, |size| len.min(size));
        self.io.issue(IoTarget::Range {
            offset: size - len as u64,
            len,
        })
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
        let Phase::Deserialising { size, deserializer } = &mut self.phase else {
            vortex_bail!("FooterOpen: deserialize called outside the deserialising phase");
        };
        let size = *size;
        match deserializer.deserialize()? {
            DeserializeStep::NeedMoreData { offset, len } => Ok(PlannerOutput::NeedsIO(
                self.io.issue(IoTarget::Range { offset, len }),
            )),
            DeserializeStep::NeedFileSize => {
                vortex_bail!(
                    "FooterOpen: deserializer asked for the file size after it was supplied"
                )
            }
            DeserializeStep::Done(footer) => self.emit(size, footer),
        }
    }

    fn take_size(&mut self) -> VortexResult<u64> {
        match self.io.take() {
            Some(IoResult::Size(size)) => Ok(size),
            _ => vortex_bail!("FooterOpen: compute called before the size was delivered"),
        }
    }

    fn take_bytes(&mut self) -> VortexResult<Option<ByteBuffer>> {
        match self.io.take() {
            Some(IoResult::Bytes(handle)) => Ok(Some(handle.try_into_host_sync()?)),
            Some(IoResult::Size(_)) => vortex_bail!("FooterOpen: expected bytes, found a size"),
            None => Ok(None),
        }
    }
}

impl IoConsumer for FooterOpen {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.io.deliver(request, result);
    }
}

impl Planner for FooterOpen {
    fn state(&self) -> State {
        if matches!(self.phase, Phase::Emitted) {
            return State::Done;
        }
        match self.io.batch() {
            Some(batch) => State::NeedsIO(batch),
            None => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        match std::mem::replace(&mut self.phase, Phase::Emitted) {
            Phase::Cached { size, footer } => self.emit(size, footer),
            Phase::NeedSize { footer } => {
                let size = self.take_size()?;
                match footer {
                    Some(footer) => self.emit(size, footer),
                    None => {
                        let batch = self.issue_tail(size);
                        self.phase = Phase::NeedTail { size };
                        Ok(PlannerOutput::NeedsIO(batch))
                    }
                }
            }
            Phase::NeedTail { size } => {
                let Some(tail) = self.take_bytes()? else {
                    vortex_bail!("FooterOpen: compute called before the tail was delivered");
                };
                self.phase = Phase::Deserialising {
                    size,
                    deserializer: Footer::deserializer(tail, self.session.clone()).with_size(size),
                };
                self.deserialize()
            }
            Phase::Deserialising {
                size,
                mut deserializer,
            } => {
                if let Some(more) = self.take_bytes()? {
                    deserializer.prefix_data(more);
                }
                self.phase = Phase::Deserialising { size, deserializer };
                self.deserialize()
            }
            Phase::Emitted => vortex_bail!("FooterOpen: compute called after Done"),
        }
    }
}

#[cfg(test)]
mod tests;
