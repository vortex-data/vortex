// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Older vectorized dataflow prototype pieces for runtime information.
//!
//! The live scheduler bridge is in [`crate::v2::scheduler`]. This module keeps
//! the lower-level demand/coverage model that the scheduler experiments still use.

#![allow(dead_code)]

use std::ops::Range;

use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::v2::domain::DomainId;

/// How much runtime information is known for a requested range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Coverage {
    /// Nothing is known for the requested range.
    Unknown,
    /// Some, but not all, positions are known.
    Partial { covered_rows: u64, total_rows: u64 },
    /// Every position in the requested range is known.
    Complete,
}

impl Coverage {
    /// True when all requested positions are covered.
    pub(crate) fn is_complete(self) -> bool {
        matches!(self, Coverage::Complete)
    }
}

/// A sorted, non-overlapping set of covered ordinal ranges.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct RangeSet {
    ranges: Vec<Range<u64>>,
}

impl RangeSet {
    fn insert(&mut self, range: Range<u64>) {
        if range.start >= range.end {
            return;
        }

        let mut next_start = range.start;
        let mut next_end = range.end;
        let mut output = Vec::with_capacity(self.ranges.len() + 1);
        let mut inserted = false;

        for existing in self.ranges.drain(..) {
            if existing.end < next_start {
                output.push(existing);
            } else if next_end < existing.start {
                if !inserted {
                    output.push(next_start..next_end);
                    inserted = true;
                }
                output.push(existing);
            } else {
                next_start = next_start.min(existing.start);
                next_end = next_end.max(existing.end);
            }
        }

        if !inserted {
            output.push(next_start..next_end);
        }
        self.ranges = output;
    }

    fn coverage(&self, target: &Range<u64>) -> Coverage {
        let total_rows = target.end.saturating_sub(target.start);
        if total_rows == 0 {
            return Coverage::Complete;
        }

        let covered_rows = self.covered_rows(target);
        if covered_rows == 0 {
            Coverage::Unknown
        } else if covered_rows == total_rows {
            Coverage::Complete
        } else {
            Coverage::Partial {
                covered_rows,
                total_rows,
            }
        }
    }

    fn covered_rows(&self, target: &Range<u64>) -> u64 {
        self.ranges
            .iter()
            .map(|range| {
                let start = target.start.max(range.start);
                let end = target.end.min(range.end);
                end.saturating_sub(start)
            })
            .sum()
    }

    fn first_gap(&self, target: &Range<u64>) -> Option<Range<u64>> {
        if target.start >= target.end {
            return None;
        }

        let mut cursor = target.start;
        for range in &self.ranges {
            if range.end <= cursor {
                continue;
            }
            if range.start > cursor {
                return Some(cursor..range.start.min(target.end));
            }
            cursor = cursor.max(range.end);
            if cursor >= target.end {
                return None;
            }
        }

        (cursor < target.end).then_some(cursor..target.end)
    }

    fn covered_prefix(&self, target: &Range<u64>) -> Range<u64> {
        if target.start >= target.end {
            return target.start..target.start;
        }

        let mut cursor = target.start;
        for range in &self.ranges {
            if range.end <= cursor {
                continue;
            }
            if range.start > cursor {
                break;
            }
            cursor = cursor.max(range.end).min(target.end);
            if cursor >= target.end {
                break;
            }
        }
        target.start..cursor
    }
}

/// Exact ordinal demand with explicit coverage state.
///
/// Unknown rows are still demanded for correctness. Coverage exists so
/// schedulers can decide whether to wait for a better answer.
#[derive(Clone, Debug)]
pub(crate) struct OrdinalDemand {
    domain: DomainId,
    total_rows: u64,
    covered: RangeSet,
    masks: Vec<(Range<u64>, Mask)>,
    version: u64,
}

impl OrdinalDemand {
    /// Create an empty demand resource for an ordinal domain.
    pub(crate) fn new(domain: DomainId, total_rows: u64) -> Self {
        Self {
            domain,
            total_rows,
            covered: RangeSet::default(),
            masks: Vec::new(),
            version: 0,
        }
    }

    /// Domain this demand resource describes.
    pub(crate) fn domain(&self) -> DomainId {
        self.domain
    }

    /// Monotonic version bumped on every publication.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// Coverage for a range. This says whether demand is known, not
    /// whether rows are true.
    pub(crate) fn coverage(&self, range: &Range<u64>) -> VortexResult<Coverage> {
        self.check_range(range)?;
        Ok(self.covered.coverage(range))
    }

    /// First not-yet-covered range inside `target`.
    pub(crate) fn first_gap(&self, target: &Range<u64>) -> VortexResult<Option<Range<u64>>> {
        self.check_range(target)?;
        Ok(self.covered.first_gap(target))
    }

    /// Covered prefix of `target`.
    pub(crate) fn covered_prefix(&self, target: &Range<u64>) -> VortexResult<Range<u64>> {
        self.check_range(target)?;
        Ok(self.covered.covered_prefix(target))
    }

    /// Publish exact demand for a row range.
    pub(crate) fn publish(&mut self, range: Range<u64>, mask: Mask) -> VortexResult<()> {
        self.check_range(&range)?;
        let expected_len = usize::try_from(range.end - range.start)?;
        if mask.len() != expected_len {
            vortex_bail!(
                "published mask length {} did not match range {range:?}",
                mask.len()
            );
        }
        self.covered.insert(range.clone());
        self.masks.push((range, mask));
        self.version += 1;
        Ok(())
    }

    /// Return exact demand if the whole range is covered.
    pub(crate) fn known_mask_for(&self, range: &Range<u64>) -> VortexResult<Option<Mask>> {
        if !self.coverage(range)?.is_complete() {
            return Ok(None);
        }
        Ok(Some(self.mask_for(range)?))
    }

    /// Return the correctness mask for `range`.
    ///
    /// Unknown rows are treated as true so consumers never skip work
    /// based on missing information.
    pub(crate) fn mask_for(&self, range: &Range<u64>) -> VortexResult<Mask> {
        self.check_range(range)?;
        let len = usize::try_from(range.end - range.start)?;
        let mut bits = BitBufferMut::new_set(len);

        for (published_range, mask) in &self.masks {
            let start = range.start.max(published_range.start);
            let end = range.end.min(published_range.end);
            if start >= end {
                continue;
            }

            let output_start = usize::try_from(start - range.start)?;
            let mask_start = usize::try_from(start - published_range.start)?;
            let overlap_len = usize::try_from(end - start)?;
            for idx in 0..overlap_len {
                bits.set_to(output_start + idx, mask.value(mask_start + idx));
            }
        }

        Ok(Mask::from_buffer(bits.freeze()))
    }

    fn check_range(&self, range: &Range<u64>) -> VortexResult<()> {
        if range.start > range.end || range.end > self.total_rows {
            vortex_bail!(
                "range {range:?} exceeds ordinal domain with {} rows",
                self.total_rows
            );
        }
        Ok(())
    }
}

/// Estimated cost/value of waiting for a demand resource.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkEstimate {
    /// Cost to refine demand for one currently-unknown row.
    demand_refine_ns_per_row: f64,
    /// Cost to do downstream value/projection work for one row.
    downstream_ns_per_row: f64,
    /// Expected fraction of rows that demand will prove false.
    predicted_false_fraction: f64,
    /// Confidence in the selectivity estimate, 0..1.
    confidence: f64,
}

impl WorkEstimate {
    /// Construct a cost estimate.
    pub(crate) fn new(
        demand_refine_ns_per_row: f64,
        downstream_ns_per_row: f64,
        predicted_false_fraction: f64,
        confidence: f64,
    ) -> Self {
        Self {
            demand_refine_ns_per_row,
            downstream_ns_per_row,
            predicted_false_fraction,
            confidence,
        }
    }

    fn expected_saved_ns(self, rows: u64) -> f64 {
        rows as f64
            * self.downstream_ns_per_row.max(0.0)
            * self.predicted_false_fraction.clamp(0.0, 1.0)
            * self.confidence.clamp(0.0, 1.0)
    }

    fn expected_refine_ns(self, rows: u64) -> f64 {
        rows as f64 * self.demand_refine_ns_per_row.max(0.0)
    }
}

/// Reason a scheduler granted or withheld work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PermitReason {
    /// Drive a demand/SIP producer over an uncovered range.
    DriveDemandProducer,
    /// The requested range is already covered by the demand resource.
    AlreadyCovered,
    /// Demand is known and at least one row remains live.
    ProceedWithKnownDemand,
    /// Demand is known all-false; advance coordinates without polling
    /// the value producer.
    SkipAllFalse,
    /// Waiting is expected to save more than speculative execution.
    WaitForDemand,
    /// Speculation is cheaper than waiting for better information.
    Speculate,
}

/// Work grant for a vectorized producer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WorkPermit {
    /// Coordinate range the decision applies to.
    range: Range<u64>,
    /// Number of rows the producer may poll/execute. This can be zero
    /// for wait and skip decisions.
    rows_to_poll: u64,
    /// Why the scheduler made this decision.
    reason: PermitReason,
}

impl WorkPermit {
    fn new(range: Range<u64>, rows_to_poll: u64, reason: PermitReason) -> Self {
        Self {
            range,
            rows_to_poll,
            reason,
        }
    }

    /// Range this permit covers.
    pub(crate) fn range(&self) -> &Range<u64> {
        &self.range
    }

    /// Rows the producer should poll or execute.
    pub(crate) fn rows_to_poll(&self) -> u64 {
        self.rows_to_poll
    }

    /// Reason for this permit.
    pub(crate) fn reason(&self) -> PermitReason {
        self.reason
    }
}

/// Coarse policy for the prototype scheduler.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PermitPolicy {
    /// Max rows to grant to a demand producer in one step.
    producer_rows: u64,
    /// Max rows to grant to a value producer when speculating.
    speculative_rows: u64,
    /// Require expected savings to exceed refine cost by this factor
    /// before waiting for demand.
    wait_bias: f64,
}

impl Default for PermitPolicy {
    fn default() -> Self {
        Self {
            producer_rows: 64 * 1024,
            speculative_rows: 8 * 1024,
            wait_bias: 1.0,
        }
    }
}

impl PermitPolicy {
    /// Construct a policy with explicit row budgets.
    pub(crate) fn new(producer_rows: u64, speculative_rows: u64, wait_bias: f64) -> Self {
        Self {
            producer_rows: producer_rows.max(1),
            speculative_rows: speculative_rows.max(1),
            wait_bias: wait_bias.max(0.0),
        }
    }

    /// Grant work to a demand producer for the first uncovered part
    /// of `target`.
    pub(crate) fn demand_producer_permit(
        self,
        demand: &OrdinalDemand,
        target: &Range<u64>,
    ) -> VortexResult<WorkPermit> {
        let Some(gap) = demand.first_gap(target)? else {
            return Ok(WorkPermit::new(
                target.start..target.start,
                0,
                PermitReason::AlreadyCovered,
            ));
        };
        let end = gap.end.min(gap.start + self.producer_rows);
        Ok(WorkPermit::new(
            gap.start..end,
            end - gap.start,
            PermitReason::DriveDemandProducer,
        ))
    }

    /// Decide whether a value producer should poll, wait, speculate,
    /// or skip based on demand coverage for `target`.
    pub(crate) fn value_consumer_permit(
        self,
        demand: &OrdinalDemand,
        target: &Range<u64>,
        estimate: WorkEstimate,
    ) -> VortexResult<WorkPermit> {
        let prefix = demand.covered_prefix(target)?;
        if prefix.start < prefix.end {
            let Some(mask) = demand.known_mask_for(&prefix)? else {
                vortex_bail!("covered prefix {prefix:?} did not have a known mask");
            };
            if mask.all_false() {
                return Ok(WorkPermit::new(prefix, 0, PermitReason::SkipAllFalse));
            }
            return Ok(WorkPermit::new(
                prefix.clone(),
                prefix.end - prefix.start,
                PermitReason::ProceedWithKnownDemand,
            ));
        }

        let uncovered_rows = target.end.saturating_sub(target.start);
        if uncovered_rows == 0 {
            return Ok(WorkPermit::new(
                target.start..target.start,
                0,
                PermitReason::AlreadyCovered,
            ));
        }

        let expected_saved = estimate.expected_saved_ns(uncovered_rows);
        let expected_refine = estimate.expected_refine_ns(uncovered_rows) * self.wait_bias;
        if expected_saved > expected_refine {
            return Ok(WorkPermit::new(
                target.start..target.start,
                0,
                PermitReason::WaitForDemand,
            ));
        }

        let end = target.end.min(target.start + self.speculative_rows);
        Ok(WorkPermit::new(
            target.start..end,
            end - target.start,
            PermitReason::Speculate,
        ))
    }
}
