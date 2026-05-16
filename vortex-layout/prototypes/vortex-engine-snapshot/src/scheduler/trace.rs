//! Scheduler-internal structured event trace.
//!
//! ## Why a custom trace and not `tracing`
//!
//! The scheduler emits high-volume events on its hot path — channel
//! push/pop, propagation cycles, proposal admit, broker submit — at
//! a rate that makes per-event overhead matter. We considered the
//! [`tracing`](https://docs.rs/tracing) crate and rejected it for
//! this layer for several reasons:
//!
//! - **Hot-path cost.** Every `tracing::event!` goes through dynamic
//!   subscriber dispatch, span context lookup, and field formatting.
//!   Even when the level filter rejects an event the cost is a
//!   non-trivial filter check; when accepted it's hundreds of
//!   nanoseconds. Our typed-enum push is a single tagged-union write
//!   to a per-shard ring (~10–30ns), and gating is one
//!   `AtomicBool::load` away from free.
//! - **Static event vocabulary.** Every scheduler event is known at
//!   compile time. A closed enum gives us compiler-checked
//!   exhaustiveness, lets tests `match` on variants instead of
//!   substring-matching strings, and avoids the stringly-typed
//!   field model `tracing` is built around.
//! - **Test inspection.** Today's tests assert on event presence
//!   (`trace.contains_action("run:filter")`). With a typed enum
//!   tests can keep the string-substring API for ergonomics
//!   (variants format their action label and reason on demand) and
//!   gain typed predicates when needed.
//! - **Per-shard locality.** Cross-shard contention on the trace
//!   used to be the original bottleneck (`Mutex<ScheduleTrace>` on
//!   every event). Lock-free per-shard rings are easy to bolt onto
//!   a custom enum; harder to retrofit onto `tracing`'s subscriber
//!   model.
//!
//! ## Where `tracing` *does* fit
//!
//! Operator-level / user-facing observability — "filter pruned X
//! rows," "task quiesced after N turns," "broker submitted
//! K requests" — should live in the `tracing` ecosystem and pick up
//! its sinks, span contexts, and OTLP exporters. Those events are
//! lower-volume, semantically richer, and benefit from the
//! ecosystem far more than they suffer from the dispatch overhead.
//!
//! A future bridge could emit selected `TraceEvent` variants as
//! `tracing` events when a runtime flag is set, giving production
//! observability without paying tracing cost on the hot path.
//!
//! ## Format-on-demand
//!
//! Events carry structured data, never pre-formatted strings on the
//! producer side. `Action::label()` returns a `&'static str` for
//! the action tag; `Reason::format()` returns a `String` only when
//! the consumer actually needs it (test assertions, debug
//! printing). This moves the `format!` cost out of the scheduler's
//! hot path and into the rarely-walked consume path.

use std::fmt;
use std::sync::Arc;

use crate::BrokerId;
use crate::DomainSpan;
use crate::InputPortRef;
use crate::OperatorId;
use crate::WorkClass;
use crate::brokers::LatencyClass;

/// One structured event recorded by the scheduler. Variants are
/// cheap to push: most carry only `Copy` types and at most an
/// `Arc<str>` for an operator/broker label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceEvent {
    /// An operator's `WorkProposal` was admitted and `run` was
    /// invoked. `score` is the composite priority/EV score the
    /// scheduler used to rank it.
    OperatorRun {
        operator: OperatorId,
        label: Arc<str>,
        class: WorkClass,
        score: i64,
    },
    /// An operator emitted a free-form trace string via
    /// `ctx.trace(...)`. Kept as a string for now since operators
    /// produce arbitrary debug messages; future migration may
    /// shift the most common ones to typed variants.
    OperatorMessage {
        operator: Option<OperatorId>,
        message: Arc<str>,
    },
    /// A broker submitted a coalesced request to the substrate.
    BrokerSubmit {
        broker: BrokerId,
        label: Arc<str>,
        latency: LatencyClass,
        required_rows: u64,
        score: i64,
    },
    /// `Operator::propagate_requirements` was invoked for this
    /// operator. Emitted by every propagation pass that picks the
    /// operator up via T1/T2/T3. Useful for asserting that pure
    /// transforms do not re-translate after every batch.
    PropagateRequirementsRan { operator: OperatorId },
    /// A consumer's input requirement changed on a channel.
    RequirementChanged { input: InputPortRef },
    /// An SPMC channel observed a merged-requirement update across
    /// its consumers.
    RequirementSpmcMerged,
    /// The root output requirement covers a contiguous initial
    /// `[0, rows)` range — typical of `Limit`-driven backpressure.
    RequirementRootRequired { rows: u64 },
    /// A contiguous suffix `[start, end)` of an operator's input is
    /// `NotNeeded`. The cancellation/zone-pruning pathway.
    RequirementNotNeeded { start: u64, end: u64 },
    /// An aggregate operator's `Limit` upstream sealed its input
    /// suffix as `NotNeeded`. Diagnostic for limit-cancellation.
    AggregateLimitSealed,
    /// A late-dynamic-filter operator marked its input suffix as
    /// `NotNeeded`. Diagnostic for late-bind cancellation.
    LateFilterMarkedSuffix,
    /// An async work item was submitted.
    AsyncSubmitted {
        label: Arc<str>,
        span: DomainSpan,
    },
    /// An async work item woke up after completion.
    AsyncWake {
        label: Arc<str>,
        span: DomainSpan,
    },
    /// An async work item was cancelled.
    AsyncCancelled {
        label: Arc<str>,
        span: DomainSpan,
    },
    /// A resource was published.
    ResourcePublished { id: Arc<str> },
    /// The memory arbiter shrank channel grants task-wide.
    MemoryGrantShrink,
    /// The memory arbiter grew channel grants task-wide.
    MemoryGrantGrow,
}

impl TraceEvent {
    /// Action label — the same string the legacy
    /// `(action, reason, score)` representation used. Stable for
    /// test assertions.
    pub fn action(&self) -> ActionLabel<'_> {
        ActionLabel(self)
    }

    /// Reason — formatted on demand. Returns the same string the
    /// legacy representation produced.
    pub fn reason(&self) -> ReasonLabel<'_> {
        ReasonLabel(self)
    }

    /// Score — class priority for events that ranked into a
    /// scheduling decision; otherwise 0.
    pub fn score(&self) -> i64 {
        match self {
            TraceEvent::OperatorRun { score, .. } => *score,
            TraceEvent::BrokerSubmit { score, .. } => *score,
            TraceEvent::PropagateRequirementsRan { .. }
            | TraceEvent::RequirementChanged { .. }
            | TraceEvent::RequirementSpmcMerged
            | TraceEvent::RequirementRootRequired { .. }
            | TraceEvent::RequirementNotNeeded { .. }
            | TraceEvent::AggregateLimitSealed
            | TraceEvent::LateFilterMarkedSuffix
            | TraceEvent::ResourcePublished { .. } => {
                WorkClass::PublishResource.priority()
            }
            TraceEvent::AsyncSubmitted { .. }
            | TraceEvent::AsyncWake { .. } => WorkClass::Emit.priority(),
            TraceEvent::AsyncWake { .. } => WorkClass::Emit.priority(),
            TraceEvent::AsyncCancelled { .. } => WorkClass::Release.priority(),
            TraceEvent::MemoryGrantShrink | TraceEvent::MemoryGrantGrow => {
                WorkClass::Release.priority()
            }
            TraceEvent::OperatorMessage { .. } => WorkClass::Cpu.priority(),
        }
    }
}

/// Display wrapper for an event's action label. Kept as a separate
/// type so `Display`/`AsRef<str>` callers don't pay allocation
/// when only a static action tag is needed (most variants).
pub struct ActionLabel<'a>(&'a TraceEvent);

impl<'a> fmt::Display for ActionLabel<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            TraceEvent::OperatorRun { label, .. } => write!(f, "run:{label}"),
            TraceEvent::OperatorMessage { .. } => f.write_str("operator"),
            TraceEvent::BrokerSubmit { label, .. } => write!(f, "submit:{label}"),
            TraceEvent::PropagateRequirementsRan { .. }
            | TraceEvent::RequirementChanged { .. }
            | TraceEvent::RequirementSpmcMerged
            | TraceEvent::RequirementRootRequired { .. }
            | TraceEvent::RequirementNotNeeded { .. }
            | TraceEvent::AggregateLimitSealed
            | TraceEvent::LateFilterMarkedSuffix => f.write_str("propagate_requirements"),
            TraceEvent::AsyncSubmitted { .. }
            | TraceEvent::AsyncWake { .. }
            | TraceEvent::AsyncCancelled { .. } => f.write_str("async"),
            TraceEvent::ResourcePublished { .. } => f.write_str("resource"),
            TraceEvent::MemoryGrantShrink | TraceEvent::MemoryGrantGrow => {
                f.write_str("memory")
            }
        }
    }
}

/// Display wrapper for an event's reason. Formats on demand.
pub struct ReasonLabel<'a>(&'a TraceEvent);

impl<'a> fmt::Display for ReasonLabel<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            TraceEvent::OperatorRun { class, score, .. } => {
                write!(f, "{} score={}", class.label(), score)
            }
            TraceEvent::OperatorMessage { message, .. } => f.write_str(message),
            TraceEvent::BrokerSubmit {
                latency,
                required_rows,
                score,
                ..
            } => write!(
                f,
                "broker latency={latency:?} required={required_rows} score={score}"
            ),
            TraceEvent::RequirementChanged { input } => write!(
                f,
                "requirement changed on input {}:{}",
                input.operator().index(),
                input.port().index()
            ),
            TraceEvent::RequirementSpmcMerged => f.write_str("spmc merged requirement"),
            TraceEvent::RequirementRootRequired { rows } => {
                write!(f, "root requirement required rows [0, {rows})")
            }
            TraceEvent::RequirementNotNeeded { start, end } => {
                write!(f, "not-needed rows [{start}, {end})")
            }
            TraceEvent::AggregateLimitSealed => f.write_str("group limit sealed input suffix"),
            TraceEvent::LateFilterMarkedSuffix => {
                f.write_str("late dynamic filter marked suffix not-needed")
            }
            TraceEvent::AsyncSubmitted { label, span } => write!(
                f,
                "async submitted {label} [{}, {})",
                span.start(),
                span.end()
            ),
            TraceEvent::AsyncWake { label, span } => write!(
                f,
                "async wake {label} [{}, {})",
                span.start(),
                span.end()
            ),
            TraceEvent::AsyncCancelled { label, span } => write!(
                f,
                "async cancelled {label} [{}, {})",
                span.start(),
                span.end()
            ),
            TraceEvent::ResourcePublished { id } => write!(f, "resource {id} ready"),
            TraceEvent::MemoryGrantShrink => f.write_str("memory grant shrink"),
            TraceEvent::MemoryGrantGrow => f.write_str("memory grant grow"),
            TraceEvent::PropagateRequirementsRan { operator } => {
                write!(f, "propagate_requirements ran on op {}", operator.index())
            }
        }
    }
}

/// Captured event log. Producer side is `push`; consumers (tests,
/// debug printing) walk via `events()` or query via the
/// `contains_*` / `index_of` helpers.
///
/// Today's storage is `Vec<TraceEvent>` behind a single mutex —
/// good enough now that the event push is allocation-free for
/// most variants. The lock-free per-shard ring upgrade swaps the
/// `Vec` for `[crossbeam_queue::ArrayQueue<TraceEvent>; n_shards]`
/// without changing the producer or consumer API.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScheduleTrace {
    events: Vec<TraceEvent>,
}

impl ScheduleTrace {
    pub fn push(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// True if any event's action label contains `needle`.
    /// Backwards compat with the previous string-grep API.
    pub fn contains_action(&self, needle: &str) -> bool {
        self.events
            .iter()
            .any(|event| event.action().to_string().contains(needle))
    }

    /// True if any event's reason contains `needle`.
    pub fn contains_reason(&self, needle: &str) -> bool {
        self.events
            .iter()
            .any(|event| event.reason().to_string().contains(needle))
    }

    /// First event index whose action label contains `needle`.
    pub fn index_of(&self, needle: &str) -> Option<usize> {
        self.events
            .iter()
            .position(|event| event.action().to_string().contains(needle))
    }

    /// Count of `propagate_requirements` invocations recorded in
    /// this trace. Tests use this to assert that pure-of-output
    /// transforms are not retranslated after every batch.
    pub fn propagate_requirements_count(&self) -> usize {
        self.events
            .iter()
            .filter(|event| matches!(event, TraceEvent::PropagateRequirementsRan { .. }))
            .count()
    }

    /// Count of `propagate_requirements` invocations on the given
    /// operator.
    pub fn propagate_requirements_count_for(&self, target: OperatorId) -> usize {
        self.events
            .iter()
            .filter(|event| {
                matches!(event, TraceEvent::PropagateRequirementsRan { operator } if *operator == target)
            })
            .count()
    }

    /// True if any event matches `predicate`. Useful for typed
    /// variant assertions in tests.
    pub fn any<F: Fn(&TraceEvent) -> bool>(&self, predicate: F) -> bool {
        self.events.iter().any(predicate)
    }

    /// First event matching `predicate`, by index.
    pub fn position<F: Fn(&TraceEvent) -> bool>(&self, predicate: F) -> Option<usize> {
        self.events.iter().position(predicate)
    }
}
