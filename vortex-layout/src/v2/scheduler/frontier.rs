// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Output-frontier grants and child-specific production limits.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::future::poll_fn;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::dataflow::OrdinalDemand;
use crate::v2::domain::DomainId;
use crate::v2::domain::GrantKey;
use crate::v2::domain::SubplanId;

/// Approximate output size for a grant request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OutputEstimate {
    rows: u64,
    bytes: u64,
}

impl OutputEstimate {
    /// Construct an output estimate. `bytes` may be approximate.
    pub(crate) const fn new(rows: u64, bytes: u64) -> Self {
        Self { rows, bytes }
    }

    /// Estimated rows.
    pub(crate) fn rows(self) -> u64 {
        self.rows
    }

    /// Estimated bytes.
    pub(crate) fn bytes(self) -> u64 {
        self.bytes
    }

    fn bytes_per_row_ceil(self) -> u64 {
        if self.rows == 0 {
            0
        } else {
            self.bytes.div_ceil(self.rows).max(1)
        }
    }

    pub(crate) fn scale_to_rows(self, rows: u64) -> Self {
        if rows == self.rows {
            return self;
        }
        let bytes_per_row = self.bytes_per_row_ceil();
        Self {
            rows,
            bytes: rows.saturating_mul(bytes_per_row),
        }
    }
}

/// Request to advance a producer over a domain range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OutputGrantRequest {
    key: GrantKey,
    target: Range<u64>,
    estimate: OutputEstimate,
}

impl OutputGrantRequest {
    /// Construct a grant request.
    pub(crate) fn new(key: GrantKey, target: Range<u64>, estimate: OutputEstimate) -> Self {
        Self {
            key,
            target,
            estimate,
        }
    }

    /// Grant key.
    pub(crate) fn key(&self) -> GrantKey {
        self.key
    }

    /// Requested target range.
    pub(crate) fn target(&self) -> &Range<u64> {
        &self.target
    }

    /// Output estimate.
    pub(crate) fn estimate(&self) -> OutputEstimate {
        self.estimate
    }
}

/// Reason an output grant was or was not issued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputGrantReason {
    /// The sub-plan may produce rows up to its visible frontier.
    Granted,
    /// The request cursor is at or beyond the sub-plan's visible frontier.
    BlockedAtFrontier,
    /// The requested range is empty.
    Empty,
}

/// A bounded permission to produce output for a sub-plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OutputGrant {
    key: GrantKey,
    range: Range<u64>,
    estimate: OutputEstimate,
    visible_frontier: u64,
    reason: OutputGrantReason,
}

impl OutputGrant {
    pub(crate) fn new(
        key: GrantKey,
        range: Range<u64>,
        estimate: OutputEstimate,
        visible_frontier: u64,
        reason: OutputGrantReason,
    ) -> Self {
        Self {
            key,
            range,
            estimate,
            visible_frontier,
            reason,
        }
    }

    /// Grant key.
    pub(crate) fn key(&self) -> GrantKey {
        self.key
    }

    /// Range the sub-plan may produce.
    pub(crate) fn range(&self) -> &Range<u64> {
        &self.range
    }

    /// Estimated size of the granted range.
    pub(crate) fn estimate(&self) -> OutputEstimate {
        self.estimate
    }

    /// Visible frontier for this sub-plan at grant time.
    pub(crate) fn visible_frontier(&self) -> u64 {
        self.visible_frontier
    }

    /// Reason for the grant decision.
    pub(crate) fn reason(&self) -> OutputGrantReason {
        self.reason
    }
}

/// Per-domain, per-sub-plan frontier controller.
#[derive(Clone, Debug)]
pub(crate) struct OutputGrantor {
    domain_rows: BTreeMap<DomainId, u64>,
    frontiers: BTreeMap<GrantKey, u64>,
    default_rows_per_grant: u64,
    default_bytes_per_grant: u64,
}

impl OutputGrantor {
    /// Construct a grantor with per-grant row and byte caps.
    pub(crate) fn new(default_rows_per_grant: u64, default_bytes_per_grant: u64) -> Self {
        Self {
            domain_rows: BTreeMap::new(),
            frontiers: BTreeMap::new(),
            default_rows_per_grant: default_rows_per_grant.max(1),
            default_bytes_per_grant: default_bytes_per_grant.max(1),
        }
    }

    /// Register a bounded row domain.
    pub(crate) fn register_domain(&mut self, domain: DomainId, total_rows: u64) {
        self.domain_rows.insert(domain, total_rows);
    }

    /// Set the visible frontier for one `(domain, subplan)`.
    pub(crate) fn set_frontier(&mut self, key: GrantKey, frontier: u64) -> VortexResult<()> {
        let frontier = self.clamp_frontier(key.domain(), frontier)?;
        self.frontiers.insert(key, frontier);
        Ok(())
    }

    /// Increase the visible frontier for one `(domain, subplan)`.
    pub(crate) fn advance_frontier(&mut self, key: GrantKey, rows: u64) -> VortexResult<()> {
        let current = self.frontier(key);
        self.set_frontier(key, current.saturating_add(rows))
    }

    /// Visible frontier for one `(domain, subplan)`.
    pub(crate) fn frontier(&self, key: GrantKey) -> u64 {
        self.frontiers.get(&key).copied().unwrap_or(0)
    }

    /// Issue a bounded output grant for one sub-plan.
    pub(crate) fn grant(&self, request: OutputGrantRequest) -> VortexResult<OutputGrant> {
        self.check_range(request.key.domain(), &request.target)?;
        if request.target.start >= request.target.end {
            return Ok(OutputGrant::new(
                request.key,
                request.target.start..request.target.start,
                OutputEstimate::new(0, 0),
                self.frontier(request.key),
                OutputGrantReason::Empty,
            ));
        }

        let visible_frontier = self.frontier(request.key);
        if request.target.start >= visible_frontier {
            return Ok(OutputGrant::new(
                request.key,
                request.target.start..request.target.start,
                OutputEstimate::new(0, 0),
                visible_frontier,
                OutputGrantReason::BlockedAtFrontier,
            ));
        }

        let bytes_per_row = request.estimate.bytes_per_row_ceil();
        let rows_by_bytes = if bytes_per_row == 0 {
            self.default_rows_per_grant
        } else {
            self.default_bytes_per_grant / bytes_per_row
        }
        .max(1);

        let rows = request
            .target
            .end
            .min(visible_frontier)
            .saturating_sub(request.target.start)
            .min(self.default_rows_per_grant)
            .min(rows_by_bytes);
        let end = request.target.start + rows;
        let bytes = rows.saturating_mul(bytes_per_row);
        Ok(OutputGrant::new(
            request.key,
            request.target.start..end,
            OutputEstimate::new(rows, bytes),
            visible_frontier,
            OutputGrantReason::Granted,
        ))
    }

    fn clamp_frontier(&self, domain: DomainId, frontier: u64) -> VortexResult<u64> {
        let Some(total_rows) = self.domain_rows.get(&domain).copied() else {
            vortex_bail!("domain {domain:?} is not registered with output grantor");
        };
        Ok(frontier.min(total_rows))
    }

    fn check_range(&self, domain: DomainId, range: &Range<u64>) -> VortexResult<()> {
        let Some(total_rows) = self.domain_rows.get(&domain).copied() else {
            vortex_bail!("domain {domain:?} is not registered with output grantor");
        };
        if range.start > range.end || range.end > total_rows {
            vortex_bail!(
                "grant request range {range:?} exceeds domain {domain:?} with {total_rows} rows"
            );
        }
        Ok(())
    }
}

/// Source of output grants for a plan subtree.
pub(crate) trait FrontierSource: Send + Sync + 'static {
    /// Non-blocking grant attempt.
    fn grant_now(&self, request: OutputGrantRequest) -> VortexResult<OutputGrant>;

    /// Poll for an output grant. Sources that implement dynamic
    /// frontiers return `Pending` until the relevant frontier is
    /// released.
    fn poll_grant(
        &self,
        request: &OutputGrantRequest,
        _cx: &mut Context<'_>,
    ) -> Poll<VortexResult<OutputGrant>> {
        Poll::Ready(self.grant_now(request.clone()))
    }
}

#[derive(Debug)]
struct NoopFrontierSource;

impl FrontierSource for NoopFrontierSource {
    fn grant_now(&self, request: OutputGrantRequest) -> VortexResult<OutputGrant> {
        let rows = request.target.end.saturating_sub(request.target.start);
        let estimate = request.estimate.scale_to_rows(rows);
        let reason = if request.target.start >= request.target.end {
            OutputGrantReason::Empty
        } else {
            OutputGrantReason::Granted
        };
        Ok(OutputGrant::new(
            request.key,
            request.target.clone(),
            estimate,
            request.target.end,
            reason,
        ))
    }
}

/// Clone-cheap output frontier passed through plan execution.
///
/// The no-op frontier grants the full requested range. Future
/// ConjunctPlan specializations can wrap a different source and pass
/// child-specific keys/scopes while preserving this leaf-facing API.
#[derive(Clone)]
pub struct OutputFrontier {
    inner: Arc<dyn FrontierSource>,
    key: GrantKey,
    scope: Range<u64>,
    cursor: u64,
}

impl std::fmt::Debug for OutputFrontier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputFrontier")
            .field("key", &self.key)
            .field("scope", &self.scope)
            .field("cursor", &self.cursor)
            .finish_non_exhaustive()
    }
}

impl OutputFrontier {
    /// Construct a no-op frontier over `0..total_rows`.
    pub fn unbounded(total_rows: u64) -> Self {
        Self {
            inner: Arc::new(NoopFrontierSource),
            key: GrantKey::new(DomainId::new(0), SubplanId::new(0)),
            scope: 0..total_rows,
            cursor: 0,
        }
    }

    /// Construct a frontier from an explicit source and key.
    pub(crate) fn new(inner: Arc<dyn FrontierSource>, key: GrantKey, total_rows: u64) -> Self {
        Self {
            inner,
            key,
            scope: 0..total_rows,
            cursor: 0,
        }
    }

    /// Return a sideways clone with a different sub-plan key in the
    /// same row domain and scope. The cursor is independent and starts
    /// at zero for the cloned view.
    pub(crate) fn clone_sideways(&self, subplan: SubplanId) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            key: GrantKey::new(self.key.domain(), subplan),
            scope: self.scope.clone(),
            cursor: 0,
        }
    }

    /// Backwards-compatible alias while the prototype call sites
    /// settle on terminology.
    pub(crate) fn with_subplan(&self, subplan: SubplanId) -> Self {
        self.clone_sideways(subplan)
    }

    /// Clone with an offset into the same domain. Returned grants are
    /// expressed in the child's local coordinates and the cursor starts
    /// at zero for the child view.
    pub fn clone_with_offset(&self, sub_range: Range<u64>) -> Self {
        let global_start = self.scope.start + sub_range.start;
        let global_end = self.scope.start + sub_range.end;
        debug_assert!(
            global_end <= self.scope.end,
            "OutputFrontier::clone_with_offset: sub_range {sub_range:?} exceeds parent total {}",
            self.total_rows()
        );
        Self {
            inner: Arc::clone(&self.inner),
            key: self.key,
            scope: global_start..global_end,
            cursor: 0,
        }
    }

    /// Backwards-compatible alias for offset cloning.
    pub(crate) fn scope(&self, sub_range: Range<u64>) -> Self {
        self.clone_with_offset(sub_range)
    }

    /// Total rows visible through this frontier view.
    pub(crate) fn total_rows(&self) -> u64 {
        self.scope.end - self.scope.start
    }

    /// Request permission to produce up to `max_rows` starting at this
    /// view's current cursor. The cursor advances by the granted
    /// rows. A blocked grant advances by zero, so retrying after the
    /// frontier moves asks the same question again.
    pub(crate) fn grant_next(
        &mut self,
        max_rows: u64,
        estimate: OutputEstimate,
    ) -> VortexResult<OutputGrant> {
        let end = self.total_rows().min(self.cursor.saturating_add(max_rows));
        let grant = self.grant(self.cursor..end, estimate)?;
        debug_assert_eq!(
            grant.range.start, self.cursor,
            "frontier grants must be sequential for grant_next"
        );
        self.cursor = grant.range.end;
        Ok(grant)
    }

    /// Async grant request for dynamic frontier sources. This is the
    /// leaf-facing backpressure API: a leaf asks how much of the next
    /// `max_rows` it may produce, and waits until the parent releases
    /// enough frontier.
    pub(crate) async fn grant_next_async(
        &mut self,
        max_rows: u64,
        estimate: OutputEstimate,
    ) -> VortexResult<OutputGrant> {
        let end = self.total_rows().min(self.cursor.saturating_add(max_rows));
        let range = self.cursor..end;
        let global = self.to_global(&range)?;
        let rows = global.end.saturating_sub(global.start);
        let request = OutputGrantRequest::new(self.key, global, estimate.scale_to_rows(rows));
        let grant = poll_fn(|cx| self.inner.poll_grant(&request, cx)).await?;
        let grant = self.to_local_grant(grant)?;
        debug_assert_eq!(
            grant.range.start, self.cursor,
            "frontier grants must be sequential for grant_next_async"
        );
        self.cursor = grant.range.end;
        Ok(grant)
    }

    /// Request permission to produce `range` in this frontier's local
    /// coordinates. The returned grant range is also local.
    pub(crate) fn grant(
        &self,
        range: Range<u64>,
        estimate: OutputEstimate,
    ) -> VortexResult<OutputGrant> {
        let global = self.to_global(&range)?;
        let rows = global.end.saturating_sub(global.start);
        let grant = self.inner.grant_now(OutputGrantRequest::new(
            self.key,
            global,
            estimate.scale_to_rows(rows),
        ))?;
        self.to_local_grant(grant)
    }

    fn to_global(&self, local: &Range<u64>) -> VortexResult<Range<u64>> {
        if local.start > local.end || local.end > self.total_rows() {
            vortex_bail!(
                "frontier grant range {local:?} exceeds scoped frontier with {} rows",
                self.total_rows()
            );
        }
        Ok((self.scope.start + local.start)..(self.scope.start + local.end))
    }

    fn to_local_grant(&self, grant: OutputGrant) -> VortexResult<OutputGrant> {
        if grant.range.start < self.scope.start || grant.range.end > self.scope.end {
            vortex_bail!(
                "frontier source granted range {:?} outside scope {:?}",
                grant.range,
                self.scope
            );
        }
        let visible_frontier = grant.visible_frontier.saturating_sub(self.scope.start);
        Ok(OutputGrant::new(
            grant.key,
            (grant.range.start - self.scope.start)..(grant.range.end - self.scope.start),
            grant.estimate,
            visible_frontier,
            grant.reason,
        ))
    }
}

impl FrontierSource for parking_lot::Mutex<OutputGrantor> {
    fn grant_now(&self, request: OutputGrantRequest) -> VortexResult<OutputGrant> {
        self.lock().grant(request)
    }
}

/// Frontier policy for an ordered conjunct pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ConjunctFrontierPolicy {
    /// How far the first demand-producing conjunct may run ahead.
    leader_rows: u64,
    /// How far later conjuncts may speculate beyond known upstream
    /// demand coverage.
    dependent_speculative_rows: u64,
    /// Row cap passed through to the output grantor.
    rows_per_grant: u64,
    /// Byte cap passed through to the output grantor.
    bytes_per_grant: u64,
}

impl ConjunctFrontierPolicy {
    /// Construct a conjunct frontier policy.
    pub(crate) fn new(
        leader_rows: u64,
        dependent_speculative_rows: u64,
        rows_per_grant: u64,
        bytes_per_grant: u64,
    ) -> Self {
        Self {
            leader_rows: leader_rows.max(1),
            dependent_speculative_rows,
            rows_per_grant: rows_per_grant.max(1),
            bytes_per_grant: bytes_per_grant.max(1),
        }
    }
}

impl Default for ConjunctFrontierPolicy {
    fn default() -> Self {
        Self::new(64 * 1024, 8 * 1024, 64 * 1024, 1024 * 1024)
    }
}

/// Prototype controller for releasing different child frontiers from
/// an ordered conjunct plan.
///
/// This is deliberately separate from the live stream implementation.
/// The model is:
///
/// - conjunct 0 is allowed to run ahead and publish demand;
/// - conjunct N+1 is released only to the covered prefix published by
///   conjunct N, plus a bounded speculative look-ahead;
/// - each child still requests concrete output grants, capped by rows
///   and estimated bytes.
#[derive(Clone, Debug)]
pub(crate) struct ConjunctFrontierController {
    domain: DomainId,
    total_rows: u64,
    subplans: Vec<SubplanId>,
    policy: ConjunctFrontierPolicy,
    grantor: OutputGrantor,
}

impl ConjunctFrontierController {
    /// Construct a controller over one ordinal row domain.
    pub(crate) fn new(
        domain: DomainId,
        total_rows: u64,
        subplans: Vec<SubplanId>,
        policy: ConjunctFrontierPolicy,
    ) -> VortexResult<Self> {
        if subplans.is_empty() {
            vortex_bail!("conjunct frontier controller needs at least one sub-plan");
        }

        let mut grantor = OutputGrantor::new(policy.rows_per_grant, policy.bytes_per_grant);
        grantor.register_domain(domain, total_rows);
        for subplan in &subplans {
            grantor.set_frontier(GrantKey::new(domain, *subplan), 0)?;
        }

        Ok(Self {
            domain,
            total_rows,
            subplans,
            policy,
            grantor,
        })
    }

    /// Begin driving a target range.
    ///
    /// The first conjunct receives the leader frontier. Later
    /// conjuncts receive only speculative runway until earlier
    /// conjuncts publish demand coverage.
    pub(crate) fn begin_range(&mut self, target: &Range<u64>) -> VortexResult<()> {
        self.check_range(target)?;
        if target.start >= target.end {
            return Ok(());
        }

        self.release_to_stage(0, target.start.saturating_add(self.policy.leader_rows))?;
        let dependent_frontier = target
            .start
            .saturating_add(self.policy.dependent_speculative_rows);
        for idx in 1..self.subplans.len() {
            self.release_to_stage(idx, dependent_frontier)?;
        }
        Ok(())
    }

    /// Release the next conjunct after `stage_idx` has published
    /// demand coverage.
    ///
    /// The next conjunct sees the upstream covered prefix plus
    /// bounded speculation. Later stages are not released here; they
    /// should advance only after their immediate predecessor publishes
    /// the combined demand it observed and produced.
    pub(crate) fn release_after_stage(
        &mut self,
        stage_idx: usize,
        upstream_demand: &OrdinalDemand,
        target: &Range<u64>,
    ) -> VortexResult<()> {
        self.check_stage(stage_idx)?;
        self.check_range(target)?;
        if upstream_demand.domain() != self.domain {
            vortex_bail!(
                "upstream demand domain {:?} did not match conjunct domain {:?}",
                upstream_demand.domain(),
                self.domain
            );
        }
        let Some(next_stage) = stage_idx.checked_add(1) else {
            return Ok(());
        };
        if next_stage >= self.subplans.len() {
            return Ok(());
        }

        let covered = upstream_demand.covered_prefix(target)?;
        let frontier = covered
            .end
            .saturating_add(self.policy.dependent_speculative_rows);
        self.release_to_stage(next_stage, frontier)
    }

    /// Request an output grant for a conjunct child.
    pub(crate) fn grant_for_stage(
        &self,
        stage_idx: usize,
        target: Range<u64>,
        estimate: OutputEstimate,
    ) -> VortexResult<OutputGrant> {
        self.check_stage(stage_idx)?;
        self.grantor.grant(OutputGrantRequest::new(
            self.stage_key(stage_idx),
            target,
            estimate,
        ))
    }

    /// Visible frontier for a conjunct child.
    pub(crate) fn stage_frontier(&self, stage_idx: usize) -> VortexResult<u64> {
        self.check_stage(stage_idx)?;
        Ok(self.grantor.frontier(self.stage_key(stage_idx)))
    }

    fn release_to_stage(&mut self, stage_idx: usize, frontier: u64) -> VortexResult<()> {
        self.check_stage(stage_idx)?;
        let key = self.stage_key(stage_idx);
        let frontier = frontier.min(self.total_rows);
        if frontier > self.grantor.frontier(key) {
            self.grantor.set_frontier(key, frontier)?;
        }
        Ok(())
    }

    fn stage_key(&self, stage_idx: usize) -> GrantKey {
        GrantKey::new(self.domain, self.subplans[stage_idx])
    }

    fn check_stage(&self, stage_idx: usize) -> VortexResult<()> {
        if stage_idx >= self.subplans.len() {
            vortex_bail!(
                "conjunct stage {stage_idx} out of range for {} stages",
                self.subplans.len()
            );
        }
        Ok(())
    }

    fn check_range(&self, target: &Range<u64>) -> VortexResult<()> {
        if target.start > target.end || target.end > self.total_rows {
            vortex_bail!(
                "conjunct target range {target:?} exceeds domain with {} rows",
                self.total_rows
            );
        }
        Ok(())
    }
}
