use std::collections::BTreeMap;

use crate::AsyncWorkId;
use crate::Batch;
use crate::DomainSpan;
use crate::OperatorId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FakeIoRequest {
    pub label: String,
    pub span: DomainSpan,
    pub delay_turns: usize,
    pub bytes: usize,
    pub batch: Batch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncWakeEvent {
    pub label: String,
    pub span: DomainSpan,
    pub owner: OperatorId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AsyncWorkStatus {
    Pending,
    Ready,
    Cancelled,
    Taken,
}

#[derive(Clone, Debug)]
struct AsyncWork {
    owner: OperatorId,
    label: String,
    span: DomainSpan,
    remaining_turns: usize,
    bytes: usize,
    batch: Option<Batch>,
    status: AsyncWorkStatus,
}

#[derive(Clone, Debug, Default)]
pub struct AsyncWorkSet {
    next_id: usize,
    work: BTreeMap<AsyncWorkId, AsyncWork>,
}

impl AsyncWorkSet {
    pub fn spawn(&mut self, owner: OperatorId, request: FakeIoRequest) -> AsyncWorkId {
        let id = AsyncWorkId::from_index(self.next_id);
        self.next_id += 1;
        self.work.insert(
            id,
            AsyncWork {
                owner,
                label: request.label,
                span: request.span,
                remaining_turns: request.delay_turns,
                bytes: request.bytes,
                batch: Some(request.batch),
                status: AsyncWorkStatus::Pending,
            },
        );
        id
    }

    pub fn advance(&mut self) -> Vec<AsyncWakeEvent> {
        let mut events = Vec::new();
        for work in self.work.values_mut() {
            if work.status != AsyncWorkStatus::Pending {
                continue;
            }
            if work.remaining_turns > 0 {
                work.remaining_turns -= 1;
            }
            if work.remaining_turns == 0 {
                work.status = AsyncWorkStatus::Ready;
                events.push(AsyncWakeEvent {
                    label: work.label.clone(),
                    span: work.span,
                    owner: work.owner,
                });
            }
        }
        events
    }

    pub fn has_ready_for(&self, owner: OperatorId) -> bool {
        self.work
            .values()
            .any(|work| work.owner == owner && work.status == AsyncWorkStatus::Ready)
    }

    pub fn take_completed(&mut self, id: AsyncWorkId) -> Option<Batch> {
        let work = self.work.get_mut(&id)?;
        if work.status != AsyncWorkStatus::Ready {
            return None;
        }
        work.status = AsyncWorkStatus::Taken;
        work.batch.take()
    }

    pub fn cancel(&mut self, id: AsyncWorkId) -> Option<(String, DomainSpan)> {
        let work = self.work.get_mut(&id)?;
        if !matches!(
            work.status,
            AsyncWorkStatus::Pending | AsyncWorkStatus::Ready
        ) {
            return None;
        }
        work.status = AsyncWorkStatus::Cancelled;
        work.batch = None;
        Some((work.label.clone(), work.span))
    }

    pub fn retained_bytes(&self) -> usize {
        self.work
            .values()
            .filter(|work| {
                matches!(
                    work.status,
                    AsyncWorkStatus::Pending | AsyncWorkStatus::Ready
                )
            })
            .map(|work| work.bytes)
            .sum()
    }

    /// True if any async work is still pending or ready-but-not-taken.
    /// The scheduler must not quiesce while this is true; pending work
    /// will produce a wake on a future `advance` call.
    pub fn has_pending(&self) -> bool {
        self.work
            .values()
            .any(|work| matches!(work.status, AsyncWorkStatus::Pending | AsyncWorkStatus::Ready))
    }
}
