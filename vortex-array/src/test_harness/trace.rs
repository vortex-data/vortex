// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;

/// Controls how much rule and kernel resolution detail is captured.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TraceResolution {
    /// Record only the operations that actually executed.
    #[default]
    ExecutedOnly,
    /// Also record rule and kernel attempts that matched but declined, or did not match.
    Attempts,
}

/// Options for [`trace_array_with`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TraceOptions {
    /// The amount of rule and kernel resolution detail to include.
    pub resolution: TraceResolution,
}

/// The result of a traced operation.
#[derive(Clone, Debug)]
pub struct Traced<T> {
    /// The value returned by the traced closure.
    pub output: T,
    /// A stable, snapshot-friendly rendering of optimizer and execution activity.
    pub trace: TraceDisplay,
}

/// A stable, snapshot-friendly trace.
#[derive(Clone, Debug, Default)]
pub struct TraceDisplay {
    options: TraceOptions,
    events: Vec<TraceEvent>,
}

impl Display for TraceDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hidden_events = self.hidden_events();
        let mut optimize_depth = 0usize;
        let mut wrote_event = false;

        for (idx, event) in self.events.iter().enumerate() {
            if hidden_events[idx] {
                continue;
            }

            if event.closes_before(self.options.resolution) {
                optimize_depth = optimize_depth.saturating_sub(1);
            }

            if event.is_hidden(self.options.resolution) {
                continue;
            }

            if wrote_event {
                writeln!(f)?;
            } else {
                wrote_event = true;
            }

            write_indent(
                f,
                optimize_depth + event.relative_indent(self.options.resolution, optimize_depth > 0),
            )?;
            event.fmt_line(f, self.options.resolution)?;

            if event.opens_after(self.options.resolution) {
                optimize_depth += 1;
            }
            if event.closes_after(self.options.resolution) {
                optimize_depth = optimize_depth.saturating_sub(1);
            }
        }
        Ok(())
    }
}

impl TraceDisplay {
    fn hidden_events(&self) -> Vec<bool> {
        let mut hidden = vec![false; self.events.len()];
        if self.options.resolution != TraceResolution::ExecutedOnly {
            return hidden;
        }

        let mut optimize_stack = Vec::new();
        for (idx, event) in self.events.iter().enumerate() {
            match event {
                TraceEvent::OptimizeStart { .. } => optimize_stack.push(idx),
                TraceEvent::OptimizeDone { changed, .. } => {
                    let Some(start) = optimize_stack.pop() else {
                        continue;
                    };
                    if !changed {
                        hidden[start..=idx].fill(true);
                    }
                }
                _ => {}
            }
        }
        hidden
    }
}

fn write_indent(f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
    for _ in 0..depth {
        f.write_str("  ")?;
    }
    Ok(())
}

/// Run `f` while capturing default trace output.
///
/// The default resolution records the rule rewrites, parent kernels, execution steps, and builder
/// activity that actually executed. Use [`trace_array_with`] and [`TraceResolution::Attempts`]
/// when a test needs to assert on every declined rule or kernel attempt.
pub fn trace_array<T>(f: impl FnOnce() -> VortexResult<T>) -> VortexResult<Traced<T>> {
    trace_array_with(TraceOptions::default(), f)
}

/// Run `f` while capturing trace output using `options`.
///
/// Trace capture is thread-local and intentionally does not propagate to worker threads. Nested
/// trace captures return an error so tests do not accidentally merge unrelated traces.
pub fn trace_array_with<T>(
    options: TraceOptions,
    f: impl FnOnce() -> VortexResult<T>,
) -> VortexResult<Traced<T>> {
    let interest = TraceInterest::from(options.resolution);
    ACTIVE_TRACE.with(|active| {
        let mut active = active.borrow_mut();
        if active.is_some() {
            return Err(vortex_err!("trace_array captures cannot be nested"));
        }
        *active = Some(TraceRecorder::new(options));
        Ok(())
    })?;
    TRACE_INTEREST.with(|trace_interest| trace_interest.set(interest));
    ACTIVE_TRACE_COUNT.fetch_add(1, Ordering::Relaxed);
    if interest == TraceInterest::Attempts {
        ATTEMPTS_TRACE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    let guard = ActiveTraceGuard { interest };
    let output = f();
    let recorder = ACTIVE_TRACE.with(|active| {
        active
            .borrow_mut()
            .take()
            .vortex_expect("trace recorder must be installed")
    });
    drop(guard);

    output.map(|output| Traced {
        output,
        trace: recorder.finish(),
    })
}

/// Returns true when the current thread has an active trace recorder.
#[inline(always)]
pub(crate) fn is_active() -> bool {
    if ACTIVE_TRACE_COUNT.load(Ordering::Relaxed) == 0 {
        return false;
    }
    TRACE_INTEREST.with(|interest| interest.get().is_active())
}

#[inline(always)]
fn attempts_enabled() -> bool {
    if ATTEMPTS_TRACE_COUNT.load(Ordering::Relaxed) == 0 {
        return false;
    }
    TRACE_INTEREST.with(|interest| interest.get() == TraceInterest::Attempts)
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TraceSource {
    Static,
    Session(usize),
}

impl Display for TraceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceSource::Static => f.write_str("static"),
            TraceSource::Session(idx) => write!(f, "session[{idx}]"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AttemptOutcome {
    Declined,
    NoMatch,
}

impl Display for AttemptOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttemptOutcome::Declined => f.write_str("declined"),
            AttemptOutcome::NoMatch => f.write_str("no-match"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum TraceInterest {
    #[default]
    Off,
    ExecutedOnly,
    Attempts,
}

impl TraceInterest {
    #[inline]
    fn is_active(self) -> bool {
        self != Self::Off
    }
}

impl From<TraceResolution> for TraceInterest {
    fn from(resolution: TraceResolution) -> Self {
        match resolution {
            TraceResolution::ExecutedOnly => Self::ExecutedOnly,
            TraceResolution::Attempts => Self::Attempts,
        }
    }
}

/// Snapshot-friendly wrapper around [`ArrayRef`] that renders just the encoding, length, and
/// dtype using the trace format (`vortex.primitive len=4 dtype=i32`).
///
/// Carries a clone of the [`ArrayRef`] (a cheap [`Arc`] bump) instead of cloning individual
/// fields, so trace events stay small and don't duplicate the existing array metadata.
#[derive(Clone, Debug)]
pub(crate) struct ArraySummary(ArrayRef);

impl ArraySummary {
    pub(crate) fn new(array: &ArrayRef) -> Self {
        Self(array.clone())
    }
}

impl Display for ArraySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} len={} dtype={}",
            self.0.encoding_id(),
            self.0.len(),
            self.0.dtype(),
        )
    }
}

pub(crate) fn record_optimize_start(root: &ArrayRef, session: bool) {
    record(TraceEvent::OptimizeStart {
        root: ArraySummary::new(root),
        session,
    });
}

pub(crate) fn record_optimize_loop_start(array: &ArrayRef) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::OptimizeLoopStart {
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_optimize_loop_end() {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::OptimizeLoopEnd);
}

pub(crate) fn record_optimize_reduce_none(array: &ArrayRef) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::PhaseNone {
        indent: 0,
        phase: "reduce",
        subject: "array",
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_optimize_parent_reduce_none(array: &ArrayRef) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::PhaseNone {
        indent: 0,
        phase: "reduce_parent",
        subject: "array",
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_optimize_done(output: &ArrayRef, changed: bool) {
    record(TraceEvent::OptimizeDone {
        output: ArraySummary::new(output),
        changed,
    });
}

pub(crate) fn record_optimize_recursive_start(root: &ArrayRef) {
    record(TraceEvent::OptimizeRecursiveStart {
        root: ArraySummary::new(root),
    });
}

pub(crate) fn record_optimize_recursive_slot(slot_idx: usize, input: &ArrayRef, output: &ArrayRef) {
    record(TraceEvent::OptimizeRecursiveSlot {
        slot_idx,
        input: ArraySummary::new(input),
        output: ArraySummary::new(output),
    });
}

pub(crate) fn record_reduce_attempt(array: &ArrayRef, rule: &dyn Debug, outcome: AttemptOutcome) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::ReduceAttempt {
        array: ArraySummary::new(array),
        rule: compact_label(rule),
        outcome,
    });
}

pub(crate) fn record_reduce_applied(array: &ArrayRef, rule: &dyn Debug, output: &ArrayRef) {
    record(TraceEvent::ReduceApplied {
        array: ArraySummary::new(array),
        rule: compact_label(rule),
        output: ArraySummary::new(output),
    });
}

pub(crate) fn record_parent_reduce_attempt(
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
    source: TraceSource,
    rule: impl Into<String>,
    outcome: AttemptOutcome,
) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::ParentReduceAttempt {
        parent: ArraySummary::new(parent),
        child: ArraySummary::new(child),
        slot_idx,
        source,
        rule: rule.into(),
        outcome,
    });
}

pub(crate) fn record_parent_reduce_applied(
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
    source: TraceSource,
    rule: impl Into<String>,
    output: &ArrayRef,
) {
    record(TraceEvent::ParentReduceApplied {
        parent: ArraySummary::new(parent),
        child: ArraySummary::new(child),
        slot_idx,
        source,
        rule: rule.into(),
        output: ArraySummary::new(output),
    });
}

pub(crate) fn record_execute_until_start<M>(root: &ArrayRef) {
    record(TraceEvent::ExecuteUntilStart {
        target: short_type_name::<M>(),
        root: ArraySummary::new(root),
    });
}

pub(crate) fn record_execute_until_iteration(
    iteration: usize,
    current: &ArrayRef,
    stack_parent: Option<(&ArrayRef, usize)>,
    builder_active: bool,
) {
    record(TraceEvent::ExecuteUntilIteration {
        iteration,
        current: ArraySummary::new(current),
        stack_parent: stack_parent.map(|(array, slot_idx)| (ArraySummary::new(array), slot_idx)),
        builder_active,
    });
}

pub(crate) fn record_execute_until_done_check(target: bool, canonical: bool) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::ExecuteUntilDoneCheck { target, canonical });
}

pub(crate) fn record_execute_until_return(output: &ArrayRef) {
    record(TraceEvent::ExecuteUntilReturn {
        output: ArraySummary::new(output),
    });
}

pub(crate) fn record_execute_until_pop_frame(
    parent: &ArrayRef,
    slot_idx: usize,
    child: &ArrayRef,
    output: &ArrayRef,
) {
    record(TraceEvent::ExecuteUntilPopFrame {
        parent: ArraySummary::new(parent),
        slot_idx,
        child: ArraySummary::new(child),
        output: ArraySummary::new(output),
    });
}

pub(crate) fn record_execute_parent_attempt(
    phase: &'static str,
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
    source: TraceSource,
    kernel: impl Into<String>,
    outcome: AttemptOutcome,
) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::ExecuteParentAttempt {
        phase,
        parent: ArraySummary::new(parent),
        child: ArraySummary::new(child),
        slot_idx,
        source,
        kernel: kernel.into(),
        outcome,
    });
}

pub(crate) fn record_execute_parent_applied(
    phase: &'static str,
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
    source: TraceSource,
    kernel: impl Into<String>,
    output: &ArrayRef,
) {
    record(TraceEvent::ExecuteParentApplied {
        phase,
        parent: ArraySummary::new(parent),
        child: ArraySummary::new(child),
        slot_idx,
        source,
        kernel: kernel.into(),
        output: ArraySummary::new(output),
    });
}

pub(crate) fn record_execute_parent_none(phase: &'static str, current: &ArrayRef) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::PhaseNone {
        indent: 2,
        phase,
        subject: "current",
        array: ArraySummary::new(current),
    });
}

pub(crate) fn record_execute_optimized(input: &ArrayRef, output: &ArrayRef) {
    let changed = !ArrayRef::ptr_eq(input, output);
    if !changed && !attempts_enabled() {
        return;
    }
    record(TraceEvent::ExecuteOptimized {
        input: ArraySummary::new(input),
        output: ArraySummary::new(output),
        changed,
    });
}

pub(crate) fn record_execute_encoding(array: &ArrayRef) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::ExecuteEncoding {
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_execute_step_request<M>(array: &ArrayRef, slot_idx: usize) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::ExecutionRequest {
        step: "ExecuteSlot",
        parent: ArraySummary::new(array),
        slot_idx,
        target: Some(short_type_name::<M>()),
    });
}

pub(crate) fn record_append_child_request(array: &ArrayRef, slot_idx: usize) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::ExecutionRequest {
        step: "AppendChild",
        parent: ArraySummary::new(array),
        slot_idx,
        target: None,
    });
}

pub(crate) fn record_execute_slot(slot_idx: usize, parent: &ArrayRef, child: &ArrayRef) {
    record(TraceEvent::SlotTransition {
        step: "ExecuteSlot",
        slot_idx,
        parent: ArraySummary::new(parent),
        child: ArraySummary::new(child),
    });
}

pub(crate) fn record_builder_start(array: &ArrayRef) {
    record(TraceEvent::BuilderEvent {
        action: "start",
        subject: "array",
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_append_child(slot_idx: usize, parent: &ArrayRef, child: &ArrayRef) {
    record(TraceEvent::SlotTransition {
        step: "AppendChild",
        slot_idx,
        parent: ArraySummary::new(parent),
        child: ArraySummary::new(child),
    });
}

pub(crate) fn record_builder_append(child: &ArrayRef) {
    record(TraceEvent::BuilderEvent {
        action: "append",
        subject: "child",
        array: ArraySummary::new(child),
    });
}

pub(crate) fn record_execute_done(array: &ArrayRef) {
    record(TraceEvent::ExecuteDone {
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_builder_finish(output: &ArrayRef) {
    record(TraceEvent::BuilderEvent {
        action: "finish",
        subject: "output",
        array: ArraySummary::new(output),
    });
}

pub(crate) fn record_single_step_start(array: &ArrayRef) {
    record(TraceEvent::SingleStepStart {
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_single_step_phase_none(phase: &'static str, array: &ArrayRef) {
    if !attempts_enabled() {
        return;
    }
    record(TraceEvent::PhaseNone {
        indent: 1,
        phase,
        subject: "array",
        array: ArraySummary::new(array),
    });
}

pub(crate) fn record_single_step_applied(phase: &'static str, input: &ArrayRef, output: &ArrayRef) {
    record(TraceEvent::SingleStepApplied {
        phase,
        input: ArraySummary::new(input),
        output: ArraySummary::new(output),
    });
}

pub(crate) fn with_execute_parent_phase<R>(phase: &'static str, f: impl FnOnce() -> R) -> R {
    EXECUTE_PARENT_PHASE.with(|active| {
        let previous = active.replace(phase);
        let result = f();
        active.set(previous);
        result
    })
}

pub(crate) fn with_execute_parent_phase_if_active<R>(
    phase: &'static str,
    f: impl FnOnce() -> R,
) -> R {
    if is_active() {
        with_execute_parent_phase(phase, f)
    } else {
        f()
    }
}

pub(crate) fn current_execute_parent_phase() -> &'static str {
    EXECUTE_PARENT_PHASE.with(Cell::get)
}

fn record(event: TraceEvent) {
    ACTIVE_TRACE.with(|active| {
        if let Some(recorder) = active.borrow_mut().as_mut() {
            recorder.events.push(event);
        }
    });
}

pub(crate) fn compact_label(value: &dyn Debug) -> String {
    let label = format!("{value:?}");
    if let Some(label) = adapter_field(&label, "rule") {
        return label.to_string();
    }
    if let Some(label) = adapter_field(&label, "kernel") {
        return label.to_string();
    }
    label
}

fn adapter_field<'a>(label: &'a str, field: &str) -> Option<&'a str> {
    let marker = format!("{field}: ");
    let start = label.find(&marker)? + marker.len();
    let rest = &label[start..];
    let end = rest.rfind(" }")?;
    Some(&rest[..end])
}

fn short_type_name<T>() -> String {
    std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .vortex_expect("type names are never empty")
        .to_string()
}

thread_local! {
    static TRACE_INTEREST: Cell<TraceInterest> = const { Cell::new(TraceInterest::Off) };
    static ACTIVE_TRACE: RefCell<Option<TraceRecorder>> = const { RefCell::new(None) };
    static EXECUTE_PARENT_PHASE: Cell<&'static str> = const { Cell::new("execute_parent") };
}

static ACTIVE_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
static ATTEMPTS_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);

struct ActiveTraceGuard {
    interest: TraceInterest,
}

impl Drop for ActiveTraceGuard {
    fn drop(&mut self) {
        if self.interest == TraceInterest::Attempts {
            ATTEMPTS_TRACE_COUNT.fetch_sub(1, Ordering::Relaxed);
        }
        ACTIVE_TRACE_COUNT.fetch_sub(1, Ordering::Relaxed);
        TRACE_INTEREST.with(|interest| interest.set(TraceInterest::Off));
        ACTIVE_TRACE.with(|active| {
            active.borrow_mut().take();
        });
    }
}

#[derive(Debug)]
struct TraceRecorder {
    options: TraceOptions,
    events: Vec<TraceEvent>,
}

impl TraceRecorder {
    fn new(options: TraceOptions) -> Self {
        Self {
            options,
            events: Vec::new(),
        }
    }

    fn finish(self) -> TraceDisplay {
        TraceDisplay {
            options: self.options,
            events: self.events,
        }
    }
}

#[derive(Clone, Debug)]
enum TraceEvent {
    OptimizeStart {
        root: ArraySummary,
        session: bool,
    },
    OptimizeLoopStart {
        array: ArraySummary,
    },
    OptimizeLoopEnd,
    OptimizeDone {
        output: ArraySummary,
        changed: bool,
    },
    OptimizeRecursiveStart {
        root: ArraySummary,
    },
    OptimizeRecursiveSlot {
        slot_idx: usize,
        input: ArraySummary,
        output: ArraySummary,
    },
    ReduceAttempt {
        array: ArraySummary,
        rule: String,
        outcome: AttemptOutcome,
    },
    ReduceApplied {
        array: ArraySummary,
        rule: String,
        output: ArraySummary,
    },
    ParentReduceAttempt {
        parent: ArraySummary,
        child: ArraySummary,
        slot_idx: usize,
        source: TraceSource,
        rule: String,
        outcome: AttemptOutcome,
    },
    ParentReduceApplied {
        parent: ArraySummary,
        child: ArraySummary,
        slot_idx: usize,
        source: TraceSource,
        rule: String,
        output: ArraySummary,
    },
    ExecuteUntilStart {
        target: String,
        root: ArraySummary,
    },
    ExecuteUntilIteration {
        iteration: usize,
        current: ArraySummary,
        stack_parent: Option<(ArraySummary, usize)>,
        builder_active: bool,
    },
    ExecuteUntilDoneCheck {
        target: bool,
        canonical: bool,
    },
    ExecuteUntilReturn {
        output: ArraySummary,
    },
    ExecuteUntilPopFrame {
        parent: ArraySummary,
        slot_idx: usize,
        child: ArraySummary,
        output: ArraySummary,
    },
    ExecuteParentAttempt {
        phase: &'static str,
        parent: ArraySummary,
        child: ArraySummary,
        slot_idx: usize,
        source: TraceSource,
        kernel: String,
        outcome: AttemptOutcome,
    },
    ExecuteParentApplied {
        phase: &'static str,
        parent: ArraySummary,
        child: ArraySummary,
        slot_idx: usize,
        source: TraceSource,
        kernel: String,
        output: ArraySummary,
    },
    PhaseNone {
        indent: usize,
        phase: &'static str,
        subject: &'static str,
        array: ArraySummary,
    },
    ExecuteOptimized {
        input: ArraySummary,
        output: ArraySummary,
        changed: bool,
    },
    ExecuteEncoding {
        array: ArraySummary,
    },
    ExecutionRequest {
        step: &'static str,
        parent: ArraySummary,
        slot_idx: usize,
        target: Option<String>,
    },
    SlotTransition {
        step: &'static str,
        slot_idx: usize,
        parent: ArraySummary,
        child: ArraySummary,
    },
    BuilderEvent {
        action: &'static str,
        subject: &'static str,
        array: ArraySummary,
    },
    ExecuteDone {
        array: ArraySummary,
    },
    SingleStepStart {
        array: ArraySummary,
    },
    SingleStepApplied {
        phase: &'static str,
        input: ArraySummary,
        output: ArraySummary,
    },
}

impl TraceEvent {
    fn is_hidden(&self, resolution: TraceResolution) -> bool {
        match resolution {
            TraceResolution::Attempts => matches!(self, TraceEvent::OptimizeLoopEnd),
            TraceResolution::ExecutedOnly => matches!(
                self,
                TraceEvent::OptimizeLoopStart { .. }
                    | TraceEvent::OptimizeLoopEnd
                    | TraceEvent::PhaseNone { .. }
                    | TraceEvent::ExecuteUntilDoneCheck { .. }
                    | TraceEvent::ExecuteEncoding { .. }
                    | TraceEvent::ExecutionRequest { .. }
                    | TraceEvent::ExecuteOptimized { changed: false, .. }
                    | TraceEvent::ExecuteParentAttempt { .. }
                    | TraceEvent::ReduceAttempt { .. }
                    | TraceEvent::ParentReduceAttempt { .. }
            ),
        }
    }

    fn opens_after(&self, resolution: TraceResolution) -> bool {
        match resolution {
            TraceResolution::Attempts => matches!(
                self,
                TraceEvent::OptimizeStart { .. } | TraceEvent::OptimizeLoopStart { .. }
            ),
            TraceResolution::ExecutedOnly => matches!(self, TraceEvent::OptimizeStart { .. }),
        }
    }

    fn closes_before(&self, resolution: TraceResolution) -> bool {
        match resolution {
            TraceResolution::Attempts => matches!(self, TraceEvent::OptimizeLoopEnd),
            TraceResolution::ExecutedOnly => false,
        }
    }

    fn closes_after(&self, _resolution: TraceResolution) -> bool {
        matches!(self, TraceEvent::OptimizeDone { .. })
    }

    fn relative_indent(&self, _resolution: TraceResolution, in_optimize_scope: bool) -> usize {
        match self {
            TraceEvent::OptimizeStart { .. }
            | TraceEvent::OptimizeLoopStart { .. }
            | TraceEvent::OptimizeDone { .. } => 0,
            TraceEvent::ReduceAttempt { .. }
            | TraceEvent::ReduceApplied { .. }
            | TraceEvent::ParentReduceAttempt { .. }
            | TraceEvent::ParentReduceApplied { .. }
                if in_optimize_scope =>
            {
                0
            }
            TraceEvent::PhaseNone { indent, .. } => *indent,
            TraceEvent::ReduceAttempt { .. }
            | TraceEvent::ReduceApplied { .. }
            | TraceEvent::ParentReduceAttempt { .. }
            | TraceEvent::ParentReduceApplied { .. }
            | TraceEvent::ExecuteUntilDoneCheck { .. }
            | TraceEvent::ExecuteUntilPopFrame { .. }
            | TraceEvent::ExecuteParentAttempt { .. }
            | TraceEvent::ExecuteParentApplied { .. }
            | TraceEvent::ExecuteOptimized { .. }
            | TraceEvent::ExecuteEncoding { .. }
            | TraceEvent::ExecutionRequest { .. }
            | TraceEvent::SlotTransition { .. }
            | TraceEvent::BuilderEvent { .. }
            | TraceEvent::ExecuteDone { .. } => 2,
            TraceEvent::OptimizeRecursiveSlot { .. }
            | TraceEvent::ExecuteUntilIteration { .. }
            | TraceEvent::ExecuteUntilReturn { .. }
            | TraceEvent::SingleStepApplied { .. } => 1,
            TraceEvent::OptimizeLoopEnd
            | TraceEvent::OptimizeRecursiveStart { .. }
            | TraceEvent::ExecuteUntilStart { .. }
            | TraceEvent::SingleStepStart { .. } => 0,
        }
    }

    fn fmt_line(&self, f: &mut fmt::Formatter<'_>, resolution: TraceResolution) -> fmt::Result {
        match self {
            TraceEvent::OptimizeStart { root, session } => {
                write!(f, "optimize root={root} session={session}")
            }
            TraceEvent::OptimizeLoopStart { array } => {
                write!(f, "loop input={array}")
            }
            TraceEvent::OptimizeLoopEnd => Ok(()),
            TraceEvent::OptimizeDone { output, changed } => match resolution {
                TraceResolution::Attempts => write!(f, "done output={output} changed={changed}"),
                TraceResolution::ExecutedOnly => write!(f, "done output={output}"),
            },
            TraceEvent::OptimizeRecursiveStart { root } => {
                write!(f, "optimize_recursive root={root}")
            }
            TraceEvent::OptimizeRecursiveSlot {
                slot_idx,
                input,
                output,
            } => write!(f, "recursive slot={slot_idx} input={input} output={output}"),
            TraceEvent::ReduceAttempt {
                array,
                rule,
                outcome,
            } => write!(
                f,
                "reduce attempt array={array} source=static rule={rule} outcome={outcome}"
            ),
            TraceEvent::ReduceApplied {
                array,
                rule,
                output,
            } => match resolution {
                TraceResolution::Attempts => write!(
                    f,
                    "reduce applied array={array} source=static rule={rule} output={output}"
                ),
                TraceResolution::ExecutedOnly => {
                    write!(f, "reduce {rule}: {array} -> {output}")
                }
            },
            TraceEvent::ParentReduceAttempt {
                parent,
                child,
                slot_idx,
                source,
                rule,
                outcome,
            } => write!(
                f,
                "reduce_parent attempt slot={slot_idx} parent={parent} child={child} source={source} rule={rule} outcome={outcome}"
            ),
            TraceEvent::ParentReduceApplied {
                parent,
                child,
                slot_idx,
                source,
                rule,
                output,
            } => match resolution {
                TraceResolution::Attempts => write!(
                    f,
                    "reduce_parent applied slot={slot_idx} parent={parent} child={child} source={source} rule={rule} output={output}"
                ),
                TraceResolution::ExecutedOnly => write!(
                    f,
                    "reduce_parent {source}:{rule} slot={slot_idx} parent={parent} child={child} -> {output}"
                ),
            },
            TraceEvent::ExecuteUntilStart { target, root } => {
                write!(f, "execute_until target={target} root={root}")
            }
            TraceEvent::ExecuteUntilIteration {
                iteration,
                current,
                stack_parent,
                builder_active,
            } => {
                write!(f, "iter {iteration} current={current}")?;
                if let Some((parent, slot_idx)) = stack_parent {
                    write!(f, " stack_parent={parent} slot={slot_idx}")?;
                }
                write!(f, " builder_active={builder_active}")
            }
            TraceEvent::ExecuteUntilDoneCheck { target, canonical } => {
                write!(f, "done_check target={target} canonical={canonical}")
            }
            TraceEvent::ExecuteUntilReturn { output } => {
                write!(f, "return output={output}")
            }
            TraceEvent::ExecuteUntilPopFrame {
                parent,
                slot_idx,
                child,
                output,
            } => write!(
                f,
                "pop_frame slot={slot_idx} parent={parent} child={child} output={output}"
            ),
            TraceEvent::ExecuteParentAttempt {
                phase,
                parent,
                child,
                slot_idx,
                source,
                kernel,
                outcome,
            } => write!(
                f,
                "{phase} attempt slot={slot_idx} parent={parent} child={child} source={source} kernel={kernel} outcome={outcome}"
            ),
            TraceEvent::ExecuteParentApplied {
                phase,
                parent,
                child,
                slot_idx,
                source,
                kernel,
                output,
            } => match resolution {
                TraceResolution::Attempts => write!(
                    f,
                    "{phase} applied slot={slot_idx} parent={parent} child={child} source={source} kernel={kernel} output={output}"
                ),
                TraceResolution::ExecutedOnly => write!(
                    f,
                    "{phase} {source}:{kernel} slot={slot_idx} parent={parent} child={child} -> {output}"
                ),
            },
            TraceEvent::PhaseNone {
                phase,
                subject,
                array,
                ..
            } => {
                write!(f, "{phase} none {subject}={array}")
            }
            TraceEvent::ExecuteOptimized {
                input,
                output,
                changed,
            } => match resolution {
                TraceResolution::Attempts => write!(
                    f,
                    "optimize_ctx input={input} output={output} changed={changed}"
                ),
                TraceResolution::ExecutedOnly => write!(f, "optimize_ctx {input} -> {output}"),
            },
            TraceEvent::ExecuteEncoding { array } => {
                write!(f, "execute encoding={array}")
            }
            TraceEvent::ExecutionRequest {
                step,
                parent,
                slot_idx,
                target,
            } => {
                write!(f, "request {step} slot={slot_idx}")?;
                if let Some(target) = target {
                    write!(f, " target={target}")?;
                }
                write!(f, " parent={parent}")
            }
            TraceEvent::SlotTransition {
                step,
                slot_idx,
                parent,
                child,
            } => write!(f, "{step} slot={slot_idx} parent={parent} child={child}"),
            TraceEvent::ExecuteDone { array } => {
                write!(f, "Done array={array}")
            }
            TraceEvent::BuilderEvent {
                action,
                subject,
                array,
            } => {
                write!(f, "builder {action} {subject}={array}")
            }
            TraceEvent::SingleStepStart { array } => {
                write!(f, "execute_step input={array}")
            }
            TraceEvent::SingleStepApplied {
                phase,
                input,
                output,
            } => match resolution {
                TraceResolution::Attempts => {
                    write!(f, "{phase} applied input={input} output={output}")
                }
                TraceResolution::ExecutedOnly => write!(f, "{phase} {input} -> {output}"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Filter;
    use crate::arrays::FilterArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::filter::FilterArrayExt;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::session::ArraySession;
    use crate::test_harness::trace::TraceOptions;
    use crate::test_harness::trace::TraceResolution;
    use crate::test_harness::trace::trace_array;
    use crate::test_harness::trace::trace_array_with;
    use crate::test_harness::trace_arrays::stack_parent_fixture;

    #[test]
    fn trace_optimize_reduce_fixpoint() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter([0i32, 1, 2, 3]).into_array();
        let filter =
            FilterArray::try_new(values.clone(), Mask::new_true(values.len()))?.into_array();

        let traced = trace_array(|| filter.optimize())?;

        assert!(traced.output.is::<Primitive>());
        assert_arrays_eq!(traced.output, values);
        insta::assert_snapshot!(traced.trace.to_string(), @r"
optimize root=vortex.filter len=4 dtype=i32 session=false
  reduce TrivialFilterRule: vortex.filter len=4 dtype=i32 -> vortex.primitive len=4 dtype=i32
  done output=vortex.primitive len=4 dtype=i32
");

        Ok(())
    }

    #[test]
    fn trace_optimize_parent_reduce_fixpoint_attempts() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4, 5]).into_array();
        let inner = FilterArray::try_new(
            values,
            Mask::from_iter([true, false, true, true, false, true]),
        )?
        .into_array();
        let outer =
            FilterArray::try_new(inner, Mask::from_iter([false, true, true, false]))?.into_array();

        let traced = trace_array_with(
            TraceOptions {
                resolution: TraceResolution::ExecutedOnly,
            },
            || outer.optimize(),
        )?;

        let optimized_filter = traced.output.as_::<Filter>();
        assert!(optimized_filter.child().is::<Primitive>());
        assert_arrays_eq!(traced.output, PrimitiveArray::from_iter([2i32, 3]));
        insta::assert_snapshot!(traced.trace.to_string(), @r"
optimize root=vortex.filter len=2 dtype=i32 session=false
  reduce_parent static:FilterFilterRule slot=0 parent=vortex.filter len=2 dtype=i32 child=vortex.filter len=4 dtype=i32 -> vortex.filter len=2 dtype=i32
  done output=vortex.filter len=2 dtype=i32
");

        let mut ctx = ExecutionCtx::new(VortexSession::empty().with::<ArraySession>());
        let traced = trace_array_with(
            TraceOptions {
                resolution: TraceResolution::ExecutedOnly,
            },
            || outer.execute::<Canonical>(&mut ctx),
        )?;

        insta::assert_snapshot!(traced.trace.to_string(), @r"
execute_until target=AnyCanonical root=vortex.filter len=2 dtype=i32
  iter 1 current=vortex.filter len=2 dtype=i32 builder_active=false
    ExecuteSlot slot=0 parent=vortex.filter len=2 dtype=i32 child=vortex.filter len=4 dtype=i32
  iter 2 current=vortex.filter len=4 dtype=i32 stack_parent=vortex.filter len=2 dtype=i32 slot=0 builder_active=false
    Done array=vortex.primitive len=4 dtype=i32
  iter 3 current=vortex.primitive len=4 dtype=i32 stack_parent=vortex.filter len=2 dtype=i32 slot=0 builder_active=false
    pop_frame slot=0 parent=vortex.filter len=2 dtype=i32 child=vortex.primitive len=4 dtype=i32 output=vortex.filter len=2 dtype=i32
  iter 4 current=vortex.filter len=2 dtype=i32 builder_active=false
    Done array=vortex.primitive len=2 dtype=i32
  iter 5 current=vortex.primitive len=2 dtype=i32 builder_active=false
  return output=vortex.primitive len=2 dtype=i32
");

        Ok(())
    }

    #[test]
    fn trace_execution_stack_parent_kernel_attempts() -> VortexResult<()> {
        let mut ctx = ExecutionCtx::new(VortexSession::empty().with::<ArraySession>());
        let parent = stack_parent_fixture()?;

        let traced = trace_array_with(
            TraceOptions {
                resolution: TraceResolution::Attempts,
            },
            || parent.execute::<PrimitiveArray>(&mut ctx),
        )?;

        assert_arrays_eq!(traced.output, PrimitiveArray::from_iter([1i32, 2, 3]));
        insta::assert_snapshot!(traced.trace.to_string(), @r"
execute_until target=AnyCanonical root=vortex.test.stack-parent len=3 dtype=i32
  iter 1 current=vortex.test.stack-parent len=3 dtype=i32 builder_active=false
    done_check target=false canonical=false
    child_execute_parent attempt slot=0 parent=vortex.test.stack-parent len=3 dtype=i32 child=vortex.test.stack-child len=3 dtype=i32 source=static kernel=kernel[0] outcome=declined
    child_execute_parent attempt slot=0 parent=vortex.test.stack-parent len=3 dtype=i32 child=vortex.test.stack-child len=3 dtype=i32 source=static kernel=kernel[1] outcome=declined
    child_execute_parent none current=vortex.test.stack-parent len=3 dtype=i32
    execute encoding=vortex.test.stack-parent len=3 dtype=i32
    request ExecuteSlot slot=0 target=Primitive parent=vortex.test.stack-parent len=3 dtype=i32
    ExecuteSlot slot=0 parent=vortex.test.stack-parent len=3 dtype=i32 child=vortex.test.stack-child len=3 dtype=i32
  iter 2 current=vortex.test.stack-child len=3 dtype=i32 stack_parent=vortex.test.stack-parent len=3 dtype=i32 slot=0 builder_active=false
    done_check target=false canonical=false
    stack_execute_parent attempt slot=0 parent=vortex.test.stack-parent len=3 dtype=i32 child=vortex.test.stack-child len=3 dtype=i32 source=static kernel=kernel[0] outcome=declined
    stack_execute_parent applied slot=0 parent=vortex.test.stack-parent len=3 dtype=i32 child=vortex.test.stack-child len=3 dtype=i32 source=static kernel=kernel[1] output=vortex.primitive len=3 dtype=i32
optimize root=vortex.primitive len=3 dtype=i32 session=true
  loop input=vortex.primitive len=3 dtype=i32
    reduce none array=vortex.primitive len=3 dtype=i32
    reduce_parent none array=vortex.primitive len=3 dtype=i32
  done output=vortex.primitive len=3 dtype=i32 changed=false
    optimize_ctx input=vortex.primitive len=3 dtype=i32 output=vortex.primitive len=3 dtype=i32 changed=false
  iter 3 current=vortex.primitive len=3 dtype=i32 builder_active=false
    done_check target=true canonical=true
  return output=vortex.primitive len=3 dtype=i32
");

        Ok(())
    }

    #[test]
    fn trace_execution_chunked_append_child_flow() -> VortexResult<()> {
        let chunks = vec![
            PrimitiveArray::from_iter([1i32, 2]).into_array(),
            PrimitiveArray::from_iter([3i32]).into_array(),
            PrimitiveArray::from_iter([4i32, 5]).into_array(),
        ];
        let dtype = chunks[0].dtype().clone();
        let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty().with::<ArraySession>());

        let traced = trace_array(|| {
            chunked
                .execute::<Canonical>(&mut ctx)
                .map(IntoArray::into_array)
        })?;

        assert_arrays_eq!(traced.output, PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]));
        insta::assert_snapshot!(traced.trace.to_string(), @r"
execute_until target=AnyCanonical root=vortex.chunked len=5 dtype=i32
  iter 1 current=vortex.chunked len=5 dtype=i32 builder_active=false
    builder start array=vortex.chunked len=5 dtype=i32
    AppendChild slot=1 parent=vortex.chunked len=5 dtype=i32 child=vortex.primitive len=2 dtype=i32
    builder append child=vortex.primitive len=2 dtype=i32
execute_until target=AnyCanonical root=vortex.primitive len=2 dtype=i32
  iter 1 current=vortex.primitive len=2 dtype=i32 builder_active=false
  return output=vortex.primitive len=2 dtype=i32
execute_until target=AnyCanonical root=vortex.primitive len=2 dtype=i32
  iter 1 current=vortex.primitive len=2 dtype=i32 builder_active=false
  return output=vortex.primitive len=2 dtype=i32
  iter 2 current=vortex.chunked len=5 dtype=i32 builder_active=true
    AppendChild slot=2 parent=vortex.chunked len=5 dtype=i32 child=vortex.primitive len=1 dtype=i32
    builder append child=vortex.primitive len=1 dtype=i32
execute_until target=AnyCanonical root=vortex.primitive len=1 dtype=i32
  iter 1 current=vortex.primitive len=1 dtype=i32 builder_active=false
  return output=vortex.primitive len=1 dtype=i32
execute_until target=AnyCanonical root=vortex.primitive len=1 dtype=i32
  iter 1 current=vortex.primitive len=1 dtype=i32 builder_active=false
  return output=vortex.primitive len=1 dtype=i32
  iter 3 current=vortex.chunked len=5 dtype=i32 builder_active=true
    AppendChild slot=3 parent=vortex.chunked len=5 dtype=i32 child=vortex.primitive len=2 dtype=i32
    builder append child=vortex.primitive len=2 dtype=i32
execute_until target=AnyCanonical root=vortex.primitive len=2 dtype=i32
  iter 1 current=vortex.primitive len=2 dtype=i32 builder_active=false
  return output=vortex.primitive len=2 dtype=i32
execute_until target=AnyCanonical root=vortex.primitive len=2 dtype=i32
  iter 1 current=vortex.primitive len=2 dtype=i32 builder_active=false
  return output=vortex.primitive len=2 dtype=i32
  iter 4 current=vortex.chunked len=5 dtype=i32 builder_active=true
    Done array=vortex.primitive len=0 dtype=i32
    builder finish output=vortex.primitive len=5 dtype=i32
  iter 5 current=vortex.primitive len=5 dtype=i32 builder_active=false
  return output=vortex.primitive len=5 dtype=i32
");

        Ok(())
    }
}
