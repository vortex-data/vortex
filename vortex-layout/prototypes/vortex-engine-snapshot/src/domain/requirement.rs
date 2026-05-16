use crate::DomainSpan;

/// Per-row probability that the row, once produced, will be useful
/// downstream. Carried alongside [`RowDemand`] on every interval and
/// used by EV-driven scheduling: an operator's `WorkValue` is derived
/// from the merged output requirement by accumulating
/// `rows × selectivity` across the intervals its work unit would
/// satisfy. Selectivity is independent of `RowDemand` — a row can be
/// `Required` (must produce) with selectivity 0.05 (probably wasted
/// work; deprioritize but don't skip).
///
/// Encoded as `u8` mapping linearly to `0.0..=1.0` via
/// `p_x256 / 255.0`. The `_x256` naming matches
/// `WorkValue::candidate(rows, p_needed_x256)` already in the engine.
///
/// **Defaults to fully selective (`p_x256 = 255`)** when an operator
/// hasn't published a more refined estimate. A plain
/// `RequirementSet::require_span(0..N)` from a sink seeds full
/// selectivity, matching the pre-selectivity behaviour where every
/// `Required` row contributed equally to EV.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Selectivity {
    p_x256: u8,
}

impl Selectivity {
    /// Definitely useful (`p = 1.0`).
    pub const FULL: Self = Self { p_x256: 255 };
    /// Definitely useless (`p = 0.0`). Pairs with `RowDemand::NotNeeded`.
    pub const ZERO: Self = Self { p_x256: 0 };
    /// 50/50 prior — useful when an operator knows pruning *may*
    /// happen but hasn't yet computed it (e.g.
    /// `ZoneMapOperator` before the zone-map resource publishes).
    pub const HALF: Self = Self { p_x256: 128 };

    /// Construct from a `0..=255` integer; saturating-clamped.
    pub const fn from_x256(p_x256: u8) -> Self {
        Self { p_x256 }
    }

    pub const fn p_x256(self) -> u8 {
        self.p_x256
    }

    /// Floating-point view, useful for tracing and tests. Hot paths
    /// should use `p_x256` directly.
    pub fn as_f64(self) -> f64 {
        f64::from(self.p_x256) / 255.0
    }

    /// Pessimistic merge: take the max. The producer must satisfy
    /// the most demanding consumer, and a row useful to *any* one
    /// of them is at least as useful as that consumer says.
    pub const fn max(self, other: Self) -> Self {
        if self.p_x256 >= other.p_x256 {
            self
        } else {
            other
        }
    }
}

impl Default for Selectivity {
    fn default() -> Self {
        Self::FULL
    }
}

/// Per-row demand state, propagated upstream through channels.
///
/// Lifecycle: every row starts as `Unknown` (the default for rows
/// not in any [`RequirementSet`] interval). As consumers propagate
/// their decisions, rows transition through:
///
/// ```text
///   Unknown ──► Needed ──► NotNeeded
///                  ▲          │
///                  └──────────┘
///                  (cannot reverse)
/// ```
///
/// `Unknown → Needed` happens when at least one consumer
/// affirmatively wants the row. `Needed → NotNeeded` happens only
/// when *all* registered consumers of the row's domain have agreed
/// they don't want it. **The transition is monotonic in this
/// direction only** — once a row reaches `NotNeeded`, it stays
/// there.
///
/// `Candidate` is run-ahead: the row may not be strictly required,
/// but speculation policy admits work for it under slack.
///
/// **Row demand never changes the output domain.** A producer that
/// sees `NotNeeded` rows emits a placeholder array with
/// `Batch::demand` set to false for those positions, preserving
/// span and length. Operators that *do* drop rows (`Filter`,
/// `Gather`, `Repartition`) mint a fresh output domain with a
/// witness back to input rows; that's distinct from row demand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowDemand {
    /// Not yet decided by propagation. Producers should *wait* —
    /// don't emit yet, don't decode, hope the next turn refines
    /// the requirement.
    #[default]
    Unknown,
    /// All registered consumers have agreed they don't want this
    /// row. Producers emit a placeholder with `demand=false` and
    /// skip real I/O / decode.
    NotNeeded,
    /// Speculative — admitted under slack policy. Producers treat
    /// like `Needed` for the V1 reference driver; richer policies
    /// might gate Candidate work on capacity.
    Candidate,
    /// At least one consumer wants this row. Producers emit real
    /// values with `demand=true`.
    Needed,
}

impl RowDemand {
    pub const fn priority(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::NotNeeded => 1,
            Self::Candidate => 2,
            Self::Needed => 3,
        }
    }
}

/// Half-open interval `[start, end)` carrying a uniform
/// [`RowDemand`] and [`Selectivity`]. Stored sorted in
/// [`RequirementSet::intervals`] and guaranteed non-overlapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowInterval {
    pub start: u64,
    pub end: u64,
    pub demand: RowDemand,
    pub selectivity: Selectivity,
}

impl RowInterval {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// A set of per-row demand intervals, sorted and non-overlapping.
///
/// The interval representation lets bulk demand
/// (`require_span(0..N)`) cost O(log K) where K is the number of
/// existing intervals, rather than O(N) per row. Rows not covered
/// by any interval implicitly have [`RowDemand::Unknown`].
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RequirementSet {
    intervals: Vec<RowInterval>,
}

impl Clone for RequirementSet {
    fn clone(&self) -> Self {
        Self {
            intervals: self.intervals.clone(),
        }
    }

    /// Allocation-preserving clone: reuses `self.intervals`'s
    /// backing storage when its capacity is sufficient. Saves an
    /// alloc/free per propagate cycle on hot paths like
    /// `StructAssembler` that fan a single requirement to many
    /// inputs.
    fn clone_from(&mut self, source: &Self) {
        self.intervals.clone_from(&source.intervals);
    }
}

impl RequirementSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: a set with `[0, len)` marked `Needed` at full
    /// selectivity.
    pub fn all_required(len: u64) -> Self {
        let mut set = Self::default();
        set.require_span(DomainSpan::new(0, len));
        set
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    pub fn clear(&mut self) {
        self.intervals.clear();
    }

    /// All intervals in this set, sorted by start, non-overlapping.
    pub fn intervals(&self) -> &[RowInterval] {
        &self.intervals
    }

    /// Iterate over (ordinal, demand) pairs row-by-row across every
    /// interval. Intended for tests and operators that genuinely
    /// need per-row data; bulk consumers should use `intervals()`.
    pub fn iter_rows(&self) -> impl Iterator<Item = (u64, RowDemand)> + '_ {
        self.intervals
            .iter()
            .flat_map(|iv| (iv.start..iv.end).map(move |o| (o, iv.demand)))
    }

    /// Get the demand at `ordinal` (`Unknown` if not covered).
    pub fn row(&self, ordinal: u64) -> RowDemand {
        match self.find_interval(ordinal) {
            Some(idx) => self.intervals[idx].demand,
            None => RowDemand::Unknown,
        }
    }

    /// Merge another requirement set into self. Returns true if
    /// `self` changed. Selectivity composes by max — a row useful
    /// to either set's intervals is at least as useful as the more
    /// demanding side.
    pub fn merge_from(&mut self, other: &Self) -> bool {
        let before = self.clone();
        for iv in &other.intervals {
            self.set_range(iv.start, iv.end, iv.demand, iv.selectivity);
        }
        before != *self
    }

    pub fn require_row(&mut self, ordinal: u64) {
        self.set_range(ordinal, ordinal + 1, RowDemand::Needed, Selectivity::FULL);
    }

    pub fn candidate_row(&mut self, ordinal: u64) {
        self.set_range(ordinal, ordinal + 1, RowDemand::Candidate, Selectivity::FULL);
    }

    pub fn not_needed_row(&mut self, ordinal: u64) {
        self.set_range(ordinal, ordinal + 1, RowDemand::NotNeeded, Selectivity::ZERO);
    }

    pub fn require_span(&mut self, span: DomainSpan) {
        self.set_range(span.start(), span.end(), RowDemand::Needed, Selectivity::FULL);
    }

    pub fn candidate_span(&mut self, span: DomainSpan) {
        self.set_range(
            span.start(),
            span.end(),
            RowDemand::Candidate,
            Selectivity::FULL,
        );
    }

    pub fn not_needed_span(&mut self, span: DomainSpan) {
        self.set_range(
            span.start(),
            span.end(),
            RowDemand::NotNeeded,
            Selectivity::ZERO,
        );
    }

    /// Like [`require_span`](Self::require_span) but with explicit
    /// selectivity. Use when the row range is required for
    /// correctness but the producer wants the EV scheduler to bias
    /// against it (e.g. `ZoneMapOperator` while its zone map is
    /// still being computed publishes `Required` with a 0.5 prior).
    pub fn require_span_with_selectivity(
        &mut self,
        span: DomainSpan,
        selectivity: Selectivity,
    ) {
        self.set_range(span.start(), span.end(), RowDemand::Needed, selectivity);
    }

    /// Like [`candidate_span`](Self::candidate_span) but with
    /// explicit selectivity.
    pub fn candidate_span_with_selectivity(
        &mut self,
        span: DomainSpan,
        selectivity: Selectivity,
    ) {
        self.set_range(span.start(), span.end(), RowDemand::Candidate, selectivity);
    }

    /// Count consecutive rows with `Needed` demand starting at 0.
    pub fn required_count_from_zero(&self) -> u64 {
        let mut count: u64 = 0;
        for iv in &self.intervals {
            if iv.start != count {
                break;
            }
            if iv.demand != RowDemand::Needed {
                break;
            }
            count = iv.end;
        }
        count
    }

    /// Whether any row at or after `start` has a non-Unknown
    /// demand. Used by the scheduler to decide whether to skip
    /// future-only sources.
    pub fn has_admitted_row_at_or_after(&self, start: u64) -> bool {
        for iv in &self.intervals {
            if iv.end <= start {
                continue;
            }
            if matches!(
                iv.demand,
                RowDemand::Needed | RowDemand::Candidate | RowDemand::NotNeeded
            ) {
                return true;
            }
        }
        false
    }

    // -- Internal: interval mutation -------------------------------

    fn find_interval(&self, ordinal: u64) -> Option<usize> {
        let pos = self.intervals.partition_point(|iv| iv.end <= ordinal);
        let iv = self.intervals.get(pos)?;
        if iv.start <= ordinal && ordinal < iv.end {
            Some(pos)
        } else {
            None
        }
    }

    /// Apply `demand` and `selectivity` to every row in
    /// `[start, end)`, merging with any existing intervals.
    /// `RowDemand` rises monotonically by priority;
    /// `Selectivity` composes by max. Splits and coalesces
    /// intervals as needed.
    fn set_range(
        &mut self,
        start: u64,
        end: u64,
        demand: RowDemand,
        selectivity: Selectivity,
    ) {
        if start >= end {
            return;
        }

        let first = self.intervals.partition_point(|iv| iv.end <= start);
        let last = self.intervals.partition_point(|iv| iv.start < end);

        let mut new_intervals: Vec<RowInterval> = Vec::new();
        let mut cursor = start;
        for idx in first..last {
            let iv = &self.intervals[idx];
            if cursor < iv.start {
                new_intervals.push(RowInterval {
                    start: cursor,
                    end: iv.start,
                    demand,
                    selectivity,
                });
            }
            if iv.start < start {
                new_intervals.push(RowInterval {
                    start: iv.start,
                    end: start,
                    demand: iv.demand,
                    selectivity: iv.selectivity,
                });
            }
            let mid_start = iv.start.max(start);
            let mid_end = iv.end.min(end);
            if mid_start < mid_end {
                let merged_demand = if demand.priority() > iv.demand.priority() {
                    demand
                } else {
                    iv.demand
                };
                let merged_selectivity = selectivity.max(iv.selectivity);
                new_intervals.push(RowInterval {
                    start: mid_start,
                    end: mid_end,
                    demand: merged_demand,
                    selectivity: merged_selectivity,
                });
            }
            if iv.end > end {
                new_intervals.push(RowInterval {
                    start: end,
                    end: iv.end,
                    demand: iv.demand,
                    selectivity: iv.selectivity,
                });
            }
            cursor = iv.end.max(cursor);
        }
        if cursor < end {
            new_intervals.push(RowInterval {
                start: cursor,
                end,
                demand,
                selectivity,
            });
        }

        let prev_idx = first.checked_sub(1);
        if let Some(prev) = prev_idx
            && let (Some(prev_iv), Some(first_new)) =
                (self.intervals.get(prev), new_intervals.first_mut())
            && prev_iv.end == first_new.start
            && prev_iv.demand == first_new.demand
            && prev_iv.selectivity == first_new.selectivity
        {
            first_new.start = prev_iv.start;
            self.splice_intervals(prev, last, new_intervals);
            return;
        }
        if let Some(last_new) = new_intervals.last_mut()
            && let Some(after) = self.intervals.get(last)
            && last_new.end == after.start
            && last_new.demand == after.demand
            && last_new.selectivity == after.selectivity
        {
            last_new.end = after.end;
            self.splice_intervals(first, last + 1, new_intervals);
            return;
        }

        self.splice_intervals(first, last, new_intervals);
    }

    fn splice_intervals(&mut self, from: usize, to: usize, replacement: Vec<RowInterval>) {
        let cleaned: Vec<_> = replacement
            .into_iter()
            .filter(|iv| iv.start < iv.end)
            .collect();
        self.intervals.splice(from..to, cleaned);
        self.coalesce_around(from);
    }

    fn coalesce_around(&mut self, idx: usize) {
        let mut start = idx.saturating_sub(1);
        while start + 1 < self.intervals.len() {
            let (left_end, left_demand, left_sel) = {
                let l = &self.intervals[start];
                (l.end, l.demand, l.selectivity)
            };
            let next = &self.intervals[start + 1];
            if left_end == next.start
                && left_demand == next.demand
                && left_sel == next.selectivity
            {
                let new_end = next.end;
                self.intervals[start].end = new_end;
                self.intervals.remove(start + 1);
            } else {
                start += 1;
            }
        }
    }
}
