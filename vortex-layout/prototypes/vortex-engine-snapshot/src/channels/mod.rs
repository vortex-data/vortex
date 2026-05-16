mod batch;

pub use batch::*;

use std::collections::BTreeSet;
use std::collections::VecDeque;

use vortex_array::expr::Expression;

use crate::EngineError;
use crate::EngineResult;
use crate::InputPortRef;
use crate::OperatorId;
use crate::RequirementSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelTopology {
    Spsc,
    Mpsc,
    Spmc,
    Mpmc,
}

impl ChannelTopology {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Spsc => "spsc",
            Self::Mpsc => "mpsc",
            Self::Spmc => "spmc",
            Self::Mpmc => "mpmc",
        }
    }

    pub fn classify(producer_count: usize, consumer_count: usize) -> Self {
        match (producer_count.max(1), consumer_count.max(1)) {
            (1, 1) => Self::Spsc,
            (_, 1) => Self::Mpsc,
            (1, _) => Self::Spmc,
            (_, _) => Self::Mpmc,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillPolicy {
    Never,
    Allowed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelBuffer {
    Inline,
    Bounded {
        min_bytes: usize,
        target_bytes: usize,
        max_bytes: usize,
    },
    Materialized {
        min_bytes: usize,
        target_bytes: usize,
        max_bytes: Option<usize>,
        spill: SpillPolicy,
    },
}

impl ChannelBuffer {
    pub const fn bounded(target_bytes: usize) -> Self {
        Self::bounded_bytes(target_bytes)
    }

    pub const fn bounded_bytes(target_bytes: usize) -> Self {
        Self::Bounded {
            min_bytes: target_bytes,
            target_bytes,
            max_bytes: target_bytes,
        }
    }

    pub const fn dynamic_bytes(min_bytes: usize, target_bytes: usize, max_bytes: usize) -> Self {
        Self::Bounded {
            min_bytes,
            target_bytes,
            max_bytes,
        }
    }

    pub const fn min_bytes(&self) -> usize {
        match self {
            Self::Inline => 0,
            Self::Bounded { min_bytes, .. } => *min_bytes,
            Self::Materialized { min_bytes, .. } => *min_bytes,
        }
    }

    pub const fn target_bytes(&self) -> usize {
        match self {
            Self::Inline => 0,
            Self::Bounded { target_bytes, .. } => *target_bytes,
            Self::Materialized { target_bytes, .. } => *target_bytes,
        }
    }

    pub const fn max_bytes(&self) -> usize {
        match self {
            Self::Inline => 0,
            Self::Bounded { max_bytes, .. } => *max_bytes,
            Self::Materialized { max_bytes, .. } => match max_bytes {
                Some(max_bytes) => *max_bytes,
                None => usize::MAX / 2,
            },
        }
    }
}

/// A channel spec records the producers feeding the channel and the
/// consumer input ports draining it. A channel has at least one
/// producer; multi-producer channels carry the union of those
/// producers' batches with arrival-order drain semantics.
///
/// **Spans are producer-domain values and the channel does not
/// rewrite them.** Operators that need a unified output domain
/// across multiple producers (e.g. `Concat`) materialise that
/// domain themselves — that's what makes them an operator. The
/// channel is a typed pipe; it transports batches verbatim.
#[derive(Clone)]
pub struct ChannelSpec {
    pub label: String,
    /// Producer operators. Length ≥ 1. Multi-producer channels
    /// subsume what the `Union` operator did previously.
    pub from: Vec<OperatorId>,
    pub to: Vec<InputPortRef>,
    pub topology: ChannelTopology,
    pub buffer: ChannelBuffer,
    /// Optional projection expression applied to every pushed
    /// batch's array on its way into the channel. Equivalent to
    /// a "Project operator" that disappears — projection is just
    /// data attached to the channel, not a separate node. Applied
    /// uniformly to every producer's pushes; if different producers
    /// need different projections the planner uses separate channels.
    pub projection: Option<Expression>,
}

impl std::fmt::Debug for ChannelSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelSpec")
            .field("label", &self.label)
            .field("from", &self.from)
            .field("to", &self.to)
            .field("topology", &self.topology)
            .field("buffer", &self.buffer)
            .field(
                "projection",
                &self.projection.as_ref().map(|e| e.to_string()),
            )
            .finish()
    }
}

impl ChannelSpec {
    pub fn single_producer(
        label: impl Into<String>,
        from: OperatorId,
        to: Vec<InputPortRef>,
        buffer: ChannelBuffer,
    ) -> Self {
        let consumer_count = to.len().max(1);
        Self {
            label: label.into(),
            from: vec![from],
            to,
            topology: ChannelTopology::classify(1, consumer_count),
            buffer,
            projection: None,
        }
    }

    pub fn multi_producer(
        label: impl Into<String>,
        from: Vec<OperatorId>,
        to: Vec<InputPortRef>,
        buffer: ChannelBuffer,
    ) -> Self {
        let producer_count = from.len().max(1);
        let consumer_count = to.len().max(1);
        Self {
            label: label.into(),
            from,
            to,
            topology: ChannelTopology::classify(producer_count, consumer_count),
            buffer,
            projection: None,
        }
    }

    /// Attach a projection expression. Every push runs `array.apply(&expr)`
    /// before the batch lands in the channel. Equivalent to inserting a
    /// "Project operator" between the producer and the consumer — but
    /// it's just data, no operator node.
    pub fn with_projection(mut self, expression: Expression) -> Self {
        self.projection = Some(expression);
        self
    }

    pub fn producer_count(&self) -> usize {
        self.from.len()
    }

    /// Position of `op` in `from`, or `None` if `op` isn't a producer.
    pub fn producer_index(&self, op: OperatorId) -> Option<usize> {
        self.from.iter().position(|p| *p == op)
    }
}

#[derive(Clone, Debug)]
struct ChannelEntry {
    batch: Batch,
    remaining: BTreeSet<InputPortRef>,
}

#[derive(Clone, Debug)]
pub struct Channel {
    spec: ChannelSpec,
    entries: VecDeque<ChannelEntry>,
    /// One flag per producer (parallels `spec.from`). The channel is
    /// fully sealed once every flag is `true`.
    producer_sealed: Vec<bool>,
    current_capacity_bytes: usize,
    consumer_requirements: Vec<RequirementSet>,
    /// Running sum of `entry.batch.estimated_bytes()` for all
    /// `entries`. Updated on push (add) and pop (subtract).
    retained_bytes_total: usize,
}

impl Channel {
    pub fn new(spec: ChannelSpec) -> Self {
        let current_capacity_bytes = spec.buffer.target_bytes();
        let consumer_requirements = vec![RequirementSet::default(); spec.to.len()];
        let producer_sealed = vec![false; spec.from.len().max(1)];
        Self {
            spec,
            entries: VecDeque::new(),
            producer_sealed,
            current_capacity_bytes,
            consumer_requirements,
            retained_bytes_total: 0,
        }
    }

    pub fn spec(&self) -> &ChannelSpec {
        &self.spec
    }

    pub const fn current_capacity(&self) -> usize {
        self.current_capacity_bytes
    }

    pub fn set_current_capacity(&mut self, capacity: usize) -> bool {
        let min = self.spec.buffer.min_bytes();
        let max = self.spec.buffer.max_bytes();
        let next = capacity.clamp(min, max);
        if next == self.current_capacity_bytes {
            return false;
        }
        self.current_capacity_bytes = next;
        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn has_capacity(&self) -> bool {
        matches!(self.spec.buffer, ChannelBuffer::Inline)
            || self.retained_bytes() < self.current_capacity_bytes
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes_total
    }

    /// True iff every producer has called `seal_from` AND the
    /// consumer at `input` has nothing left to read.
    pub fn is_finished_for(&self, input: InputPortRef) -> bool {
        self.is_fully_sealed() && self.peek(input).is_none()
    }

    pub fn is_fully_sealed(&self) -> bool {
        self.producer_sealed.iter().all(|s| *s)
    }

    /// Push a batch from `producer`. The channel optionally applies
    /// a projection expression (`spec.projection`) before the batch
    /// is enqueued. Producer-assigned spans flow through unchanged
    /// — operators that need a unified output domain across multiple
    /// producers create one themselves (see `Concat`).
    ///
    /// The producer must be a member of `spec.from`.
    pub fn push(&mut self, producer: OperatorId, batch: Batch) -> EngineResult<()> {
        let producer_idx = self.spec.producer_index(producer).ok_or_else(|| {
            EngineError::message(format!(
                "push: operator {producer:?} is not a producer of channel '{}'",
                self.spec.label
            ))
        })?;
        if self.producer_sealed[producer_idx] {
            return Err(EngineError::message(format!(
                "cannot push from sealed producer {producer:?} on channel '{}'",
                self.spec.label
            )));
        }
        if batch.len() == 0 {
            return Ok(());
        }
        // Channel-attached projection. Apply via Vortex's
        // expression engine. Identity projections (root() / "$")
        // should be detected by the planner and not attached;
        // anything attached here is non-trivial.
        let batch = if let Some(projection) = &self.spec.projection {
            let demand = batch.demand().clone();
            let span = batch.span();
            let projected = batch
                .into_array()
                .apply(projection)
                .map_err(|e| EngineError::message(format!("channel projection: {e}")))?;
            Batch::with_demand(span, projected, demand)
        } else {
            batch
        };
        if !self.has_capacity() {
            return Err(EngineError::message("channel has no capacity"));
        }
        let batch_bytes = batch.estimated_bytes();
        if !matches!(self.spec.buffer, ChannelBuffer::Inline)
            && self
                .retained_bytes_total
                .saturating_add(batch_bytes)
                > self.current_capacity_bytes
        {
            return Err(EngineError::message("channel byte capacity exceeded"));
        }
        let remaining = self.spec.to.iter().copied().collect::<BTreeSet<_>>();
        self.entries.push_back(ChannelEntry { batch, remaining });
        self.retained_bytes_total = self.retained_bytes_total.saturating_add(batch_bytes);
        Ok(())
    }

    /// Mark `producer` as having sealed its output. Returns `true` if
    /// this call transitioned the channel to fully-sealed (= every
    /// producer has now sealed).
    pub fn seal_from(&mut self, producer: OperatorId) -> EngineResult<bool> {
        let idx = self.spec.producer_index(producer).ok_or_else(|| {
            EngineError::message(format!(
                "seal_from: operator {producer:?} is not a producer of channel '{}'",
                self.spec.label
            ))
        })?;
        if self.producer_sealed[idx] {
            return Ok(false);
        }
        self.producer_sealed[idx] = true;
        Ok(self.is_fully_sealed())
    }

    pub fn peek(&self, input: InputPortRef) -> Option<&Batch> {
        self.entries
            .iter()
            .find(|entry| entry.remaining.contains(&input))
            .map(|entry| &entry.batch)
    }

    pub fn pop(&mut self, input: InputPortRef) -> Option<Batch> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.remaining.contains(&input))?;
        let batch = self.entries.get(index)?.batch.clone();
        if let Some(entry) = self.entries.get_mut(index) {
            entry.remaining.remove(&input);
        }
        self.release_consumed_front();
        Some(batch)
    }

    pub fn set_requirement(
        &mut self,
        input: InputPortRef,
        requirement: &mut RequirementSet,
    ) -> EngineResult<bool> {
        let Some(index) = self
            .spec
            .to
            .iter()
            .position(|candidate| *candidate == input)
        else {
            return Err(EngineError::message("input is not connected to channel"));
        };
        if self.consumer_requirements[index] == *requirement {
            return Ok(false);
        }
        std::mem::swap(&mut self.consumer_requirements[index], requirement);
        Ok(true)
    }

    pub fn merged_requirement(&self) -> RequirementSet {
        let mut merged = RequirementSet::default();
        for requirement in &self.consumer_requirements {
            merged.merge_from(requirement);
        }
        merged
    }

    pub fn requirement_for(&self, input: InputPortRef) -> RequirementSet {
        self.spec
            .to
            .iter()
            .position(|candidate| *candidate == input)
            .map(|index| self.consumer_requirements[index].clone())
            .unwrap_or_default()
    }

    fn release_consumed_front(&mut self) {
        while self
            .entries
            .front()
            .is_some_and(|entry| entry.remaining.is_empty())
        {
            if let Some(entry) = self.entries.pop_front() {
                self.retained_bytes_total = self
                    .retained_bytes_total
                    .saturating_sub(entry.batch.estimated_bytes());
            }
        }
    }
}

