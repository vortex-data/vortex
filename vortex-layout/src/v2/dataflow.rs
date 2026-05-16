// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental vectorized dataflow model for V2 runtime information.
//!
//! This module is intentionally isolated from the scan execution path.
//! It sketches the pieces needed to model feedback between filters,
//! row demand, and projections without committing to a full dataflow
//! runtime:
//!
//! - domain descriptors for row-ordinal and key-like identities;
//! - vectorized batches over a domain range;
//! - resources with explicit coverage/frontier state;
//! - a small permit policy that decides whether to drive demand,
//!   wait for demand, proceed with known demand, speculate, or skip.
//!
//! The goal is to prototype the shape of a "vectorized timely" model:
//! data batches flow one way, runtime information flows the other way,
//! and progress/frontiers make it explicit when information is known
//! for a range.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_stream::try_stream;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::poll_fn;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;

use crate::segments::SegmentId;
use crate::v2::demand::RowDemand;
use crate::v2::flat::SharedSegmentFuture;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::scan_ctx::ScanCtx;

/// Stable identifier for an execution domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct DomainId(u32);

impl DomainId {
    /// Construct a domain identifier.
    pub(crate) const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Stable identifier for a plan node or runtime sub-plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct SubplanId(u32);

impl SubplanId {
    /// Construct a sub-plan identifier.
    pub(crate) const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the raw identifier.
    pub(crate) const fn raw(self) -> u32 {
        self.0
    }
}

/// A grant frontier is scoped by both row domain and sub-plan.
///
/// This lets a parent show different frontiers to children over the
/// same row domain. For example, a conjunct driver can let a cheap
/// selective predicate run ahead while keeping expensive predicates
/// close to the demand frontier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct GrantKey {
    domain: DomainId,
    subplan: SubplanId,
}

impl GrantKey {
    /// Construct a grant key.
    pub(crate) const fn new(domain: DomainId, subplan: SubplanId) -> Self {
        Self { domain, subplan }
    }

    /// Domain controlled by this grant.
    pub(crate) fn domain(self) -> DomainId {
        self.domain
    }

    /// Sub-plan controlled by this grant.
    pub(crate) fn subplan(self) -> SubplanId {
        self.subplan
    }
}

/// The identity space an operator's rows live in.
///
/// `Sorted` is modeled as a separate variant in this prototype because
/// sortedness changes what can be lowered into ordinal demand. A later
/// design may represent sortedness as a property on `Keyed`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum Domain {
    /// File/partition row ordinals. This is the domain where
    /// cardinality-preserving placeholders are meaningful.
    Ordinal { id: DomainId },
    /// Rows identified by a logical key, with no ordering promise.
    Keyed { id: DomainId, key: &'static str },
    /// Rows identified by a logical key whose order can be translated
    /// back to an ordinal row domain.
    Sorted {
        id: DomainId,
        key: &'static str,
        ordinal: DomainId,
    },
}

impl Domain {
    /// Returns the ordinal domain this domain can map to exactly.
    pub(crate) fn ordinal_mapping(&self) -> Option<DomainId> {
        match self {
            Domain::Ordinal { id } => Some(*id),
            Domain::Keyed { .. } => None,
            Domain::Sorted { ordinal, .. } => Some(*ordinal),
        }
    }
}

/// A vectorized message over a contiguous domain range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VectorBatch<T> {
    domain: Domain,
    range: Range<u64>,
    payload: T,
}

impl<T> VectorBatch<T> {
    /// Construct a batch.
    pub(crate) fn new(domain: Domain, range: Range<u64>, payload: T) -> Self {
        Self {
            domain,
            range,
            payload,
        }
    }

    /// Domain carried by this batch.
    pub(crate) fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Range covered by this batch.
    pub(crate) fn range(&self) -> &Range<u64> {
        &self.range
    }

    /// Number of rows or positions covered by the batch range.
    pub(crate) fn row_count(&self) -> u64 {
        self.range.end.saturating_sub(self.range.start)
    }
}

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

/// Stable identifier for one DataFusion partition-local scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct PartitionSchedulerId(u32);

impl PartitionSchedulerId {
    /// Construct a partition scheduler identifier.
    pub(crate) const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Stable identifier for a pipeline inside one partition scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct PipelineId(usize);

impl PipelineId {
    /// Construct a pipeline identifier.
    pub(crate) const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Index into the owning scheduler's pipeline state table.
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

/// Stable identifier for a morsel.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct MorselId(u64);

impl MorselId {
    /// Construct a morsel identifier.
    pub(crate) const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Stable identifier for an I/O request tracked by the scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct IoRequestId(u64);

impl IoRequestId {
    /// Construct an I/O request identifier.
    pub(crate) const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Role a morsel plays for scheduling.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum MorselRole {
    /// Produces runtime information such as row demand or SIP filters.
    InformationProducer,
    /// Consumes information to decide whether value work is needed.
    InformationConsumer,
    /// Produces projected values.
    ValueProducer,
    /// Combines sibling streams or masks.
    Combiner,
    /// Sits on an output retirement path.
    Sink,
}

/// Source that closes a lowered pipeline.
#[derive(Clone)]
pub(crate) enum SchedulerPipelineSource {
    /// Synthetic source used by structural tests and non-segment
    /// leaves while the prototype is still abstract.
    Leaf {
        subplan: SubplanId,
        local_range: Range<u64>,
        global_range: Range<u64>,
        schema: String,
        role: MorselRole,
    },
    /// Real flat/filtered-flat segment source. The segment future
    /// itself is queued as a [`SchedulerTask::Segment`].
    Segment {
        subplan: SubplanId,
        segment_id: SegmentId,
        local_range: Range<u64>,
        global_range: Range<u64>,
        schema: String,
    },
    /// Compatibility source for the first runnable scheduler path.
    /// The plan is held in scheduler pipeline state, not in a task;
    /// tasks only carry the `PipelineId` that indexes this state.
    ExecutePlan {
        plan: LayoutPlanRef,
        row_range: Range<u64>,
        demand: RowDemand,
        frontier: OutputFrontier,
        ctx: ScanCtx,
    },
}

impl std::fmt::Debug for SchedulerPipelineSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leaf {
                subplan,
                local_range,
                global_range,
                schema,
                role,
            } => f
                .debug_struct("Leaf")
                .field("subplan", subplan)
                .field("local_range", local_range)
                .field("global_range", global_range)
                .field("schema", schema)
                .field("role", role)
                .finish(),
            Self::Segment {
                subplan,
                segment_id,
                local_range,
                global_range,
                schema,
            } => f
                .debug_struct("Segment")
                .field("subplan", subplan)
                .field("segment_id", segment_id)
                .field("local_range", local_range)
                .field("global_range", global_range)
                .field("schema", schema)
                .finish(),
            Self::ExecutePlan { row_range, .. } => f
                .debug_struct("ExecutePlan")
                .field("row_range", row_range)
                .finish_non_exhaustive(),
        }
    }
}

/// Scheduler-local state for one lowered pipeline.
#[derive(Clone, Debug)]
pub(crate) struct SchedulerPipelineState {
    source: SchedulerPipelineSource,
}

impl SchedulerPipelineState {
    fn new(source: SchedulerPipelineSource) -> Self {
        Self { source }
    }
}

/// Cost and memory estimates for one morsel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MorselEstimate {
    cpu_ns: u64,
    io_bytes: u64,
    memory_bytes: u64,
}

impl MorselEstimate {
    /// Construct a morsel estimate.
    pub(crate) const fn new(cpu_ns: u64, io_bytes: u64, memory_bytes: u64) -> Self {
        Self {
            cpu_ns,
            io_bytes,
            memory_bytes,
        }
    }

    /// Estimated queued memory.
    pub(crate) fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }
}

/// Integer version of the information-priority score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MorselPriority {
    readiness: i64,
    info_gain_ns: u64,
    cost_ns: u64,
    lead_credit: i64,
    age_credit_per_tick: i64,
    memory_pressure: i64,
    backpressure: i64,
}

impl MorselPriority {
    /// Construct a priority from the score terms.
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        readiness: i64,
        info_gain_ns: u64,
        cost_ns: u64,
        lead_credit: i64,
        age_credit_per_tick: i64,
        memory_pressure: i64,
        backpressure: i64,
    ) -> Self {
        Self {
            readiness,
            info_gain_ns,
            cost_ns,
            lead_credit,
            age_credit_per_tick,
            memory_pressure,
            backpressure,
        }
    }

    /// Priority for a demand/SIP producer that should run ahead.
    pub(crate) fn information_producer(info_gain_ns: u64, cost_ns: u64, lead_credit: i64) -> Self {
        Self::new(0, info_gain_ns, cost_ns, lead_credit, 1, 0, 0)
    }

    /// Priority for ordinary value work.
    pub(crate) fn value_work(info_gain_ns: u64, cost_ns: u64, backpressure: i64) -> Self {
        Self::new(0, info_gain_ns, cost_ns, 0, 1, 0, backpressure)
    }

    /// Score at `now_tick`, scaled by 1024 to avoid floating point
    /// ordering inside the heap.
    fn score(self, ready_tick: u64, now_tick: u64) -> i64 {
        const SCALE: u128 = 1024;
        let ratio = if self.cost_ns == 0 {
            i64::MAX / 4
        } else {
            let scaled =
                (u128::from(self.info_gain_ns).saturating_mul(SCALE)) / u128::from(self.cost_ns);
            scaled.min((i64::MAX / 4) as u128) as i64
        };
        let age = now_tick.saturating_sub(ready_tick).min(i64::MAX as u64) as i64;
        self.readiness
            .saturating_add(ratio)
            .saturating_add(self.lead_credit)
            .saturating_add(age.saturating_mul(self.age_credit_per_tick))
            .saturating_sub(self.memory_pressure)
            .saturating_sub(self.backpressure)
    }
}

/// One schedulable data morsel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerMorsel {
    id: MorselId,
    domain: DomainId,
    order_key: Range<u64>,
    role: MorselRole,
    stage: u16,
    stage_count: u16,
    estimate: MorselEstimate,
    priority: MorselPriority,
}

impl SchedulerMorsel {
    /// Construct a scheduler morsel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: MorselId,
        domain: DomainId,
        order_key: Range<u64>,
        role: MorselRole,
        stage_count: u16,
        estimate: MorselEstimate,
        priority: MorselPriority,
    ) -> Self {
        Self {
            id,
            domain,
            order_key,
            role,
            stage: 0,
            stage_count: stage_count.max(1),
            estimate,
            priority,
        }
    }

    /// Morsel id.
    pub(crate) fn id(&self) -> MorselId {
        self.id
    }

    /// Current pipeline stage.
    pub(crate) fn stage(&self) -> u16 {
        self.stage
    }

    fn advance_one_stage(&mut self) -> Option<(u16, u16)> {
        let from = self.stage;
        if self.stage + 1 >= self.stage_count {
            return None;
        }
        self.stage += 1;
        Some((from, self.stage))
    }

    fn memory_bytes(&self) -> u64 {
        self.estimate.memory_bytes()
    }
}

/// CPU/operator work tracked by the scheduler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerWorkTask {
    pipeline: PipelineId,
    morsel: SchedulerMorsel,
}

impl SchedulerWorkTask {
    /// Construct a work task.
    pub(crate) fn new(pipeline: PipelineId, morsel: SchedulerMorsel) -> Self {
        Self { pipeline, morsel }
    }

    fn memory_bytes(&self) -> u64 {
        self.morsel.memory_bytes()
    }
}

/// Segment future tracked in the same priority queue as CPU morsels.
pub(crate) struct SchedulerSegmentTask {
    id: IoRequestId,
    pipeline: PipelineId,
    segment_id: SegmentId,
    domain: DomainId,
    range: Range<u64>,
    bytes: u64,
    priority: MorselPriority,
    segment_future: Option<SharedSegmentFuture>,
}

impl std::fmt::Debug for SchedulerSegmentTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerSegmentTask")
            .field("id", &self.id)
            .field("pipeline", &self.pipeline)
            .field("segment_id", &self.segment_id)
            .field("domain", &self.domain)
            .field("range", &self.range)
            .field("bytes", &self.bytes)
            .field("has_segment_future", &self.segment_future.is_some())
            .finish_non_exhaustive()
    }
}

impl SchedulerSegmentTask {
    /// Construct a segment task.
    pub(crate) fn new(
        id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
        domain: DomainId,
        range: Range<u64>,
        bytes: u64,
        priority: MorselPriority,
        segment_future: SharedSegmentFuture,
    ) -> Self {
        Self {
            id,
            pipeline,
            segment_id,
            domain,
            range,
            bytes,
            priority,
            segment_future: Some(segment_future),
        }
    }

    #[cfg(test)]
    fn metadata_only(
        id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
        domain: DomainId,
        range: Range<u64>,
        bytes: u64,
        priority: MorselPriority,
    ) -> Self {
        Self {
            id,
            pipeline,
            segment_id,
            domain,
            range,
            bytes,
            priority,
            segment_future: None,
        }
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<VortexResult<u64>> {
        let Some(segment_future) = &mut self.segment_future else {
            return Poll::Ready(Ok(self.bytes));
        };
        match segment_future.poll_unpin(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                drop(result.map_err(VortexError::from)?);
                self.segment_future = None;
                Poll::Ready(Ok(self.bytes))
            }
        }
    }

    async fn wait(&mut self) -> VortexResult<u64> {
        poll_fn(|cx| self.poll(cx)).await
    }
}

/// Work-stealing and balancing control tasks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SchedulerControlEvent {
    /// Ask another partition scheduler for work.
    StealRequest {
        from: PartitionSchedulerId,
        target_bytes: u64,
    },
    /// Offer work to another partition scheduler.
    StealOffer {
        from: PartitionSchedulerId,
        offered_morsels: u32,
    },
    /// Recompute priorities and budgets.
    Rebalance { reason: &'static str },
}

/// Task stored in a partition-local scheduler queue.
///
/// This is deliberately an enum rather than a boxed trait object: the
/// scheduler can switch on a small set of runtime work units without a
/// per-task vtable allocation.
#[derive(Debug)]
pub(crate) enum SchedulerTask {
    /// CPU or operator work.
    Work(SchedulerWorkTask),
    /// Segment I/O future.
    Segment(SchedulerSegmentTask),
    /// Work-stealing or balancing task.
    Control(SchedulerControlEvent),
}

impl SchedulerTask {
    fn memory_bytes(&self) -> u64 {
        match self {
            SchedulerTask::Work(work) => work.memory_bytes(),
            SchedulerTask::Segment(_) | SchedulerTask::Control(_) => 0,
        }
    }

    fn priority(&self, ready_tick: u64, now_tick: u64) -> QueuePriority {
        match self {
            SchedulerTask::Work(work) => QueuePriority::new(
                event_class_for_role(work.morsel.role),
                work.morsel.priority.score(ready_tick, now_tick),
            ),
            SchedulerTask::Segment(request) => {
                QueuePriority::new(EventClass::Io, request.priority.score(ready_tick, now_tick))
            }
            SchedulerTask::Control(control) => QueuePriority::new(
                EventClass::Control,
                match control {
                    SchedulerControlEvent::Rebalance { .. } => 1_000,
                    SchedulerControlEvent::StealRequest { .. } => 500,
                    SchedulerControlEvent::StealOffer { .. } => 250,
                },
            ),
        }
    }
}

fn event_class_for_role(role: MorselRole) -> EventClass {
    match role {
        MorselRole::Sink => EventClass::RetirementCritical,
        MorselRole::InformationProducer => EventClass::Information,
        MorselRole::InformationConsumer | MorselRole::ValueProducer | MorselRole::Combiner => {
            EventClass::Data
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum EventClass {
    Control = 0,
    Io = 1,
    Data = 2,
    Information = 3,
    RetirementCritical = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QueuePriority {
    class: EventClass,
    score: i64,
}

impl QueuePriority {
    fn new(class: EventClass, score: i64) -> Self {
        Self { class, score }
    }
}

impl Ord for QueuePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.class
            .cmp(&other.class)
            .then_with(|| self.score.cmp(&other.score))
    }
}

impl PartialOrd for QueuePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct QueueEntry {
    priority: QueuePriority,
    sequence: u64,
    ready_tick: u64,
    task: SchedulerTask,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
            && self.sequence == other.sequence
            && self.ready_tick == other.ready_tick
    }
}

impl Eq for QueueEntry {}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            // BinaryHeap is max-first; reverse sequence for FIFO
            // behavior among equally-ranked events.
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Queue bounds for one partition scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerBudget {
    max_queued_events: usize,
    max_queued_memory_bytes: u64,
}

impl SchedulerBudget {
    /// Construct scheduler queue bounds.
    pub(crate) const fn new(max_queued_events: usize, max_queued_memory_bytes: u64) -> Self {
        Self {
            max_queued_events,
            max_queued_memory_bytes,
        }
    }
}

impl Default for SchedulerBudget {
    fn default() -> Self {
        Self::new(1024, 64 * 1024 * 1024)
    }
}

/// Result of letting one partition scheduler make progress once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SchedulerStep {
    /// A morsel advanced one stage and was requeued if needed.
    Advanced {
        morsel_id: MorselId,
        pipeline: PipelineId,
        from_stage: u16,
        to_stage: u16,
    },
    /// A morsel completed its last stage.
    Completed {
        morsel_id: MorselId,
        pipeline: PipelineId,
    },
    /// The scheduler completed a segment future.
    CompletedSegment {
        request_id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
        bytes: u64,
    },
    /// The scheduler polled a segment future that was not ready.
    PendingSegment {
        request_id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
    },
    /// The scheduler handled a balancing/work-stealing event.
    Control { event: SchedulerControlEvent },
}

/// Prototype partition-local scheduler.
///
/// The intended runtime shape is one scheduler per DataFusion output
/// partition. Each scheduler owns a priority queue containing data
/// morsels, I/O requests, and balancing events. Calling
/// [`Self::make_progress`] advances exactly one event.
#[derive(Debug)]
pub(crate) struct PartitionScheduler {
    id: PartitionSchedulerId,
    budget: SchedulerBudget,
    pipelines: Vec<SchedulerPipelineState>,
    queue: BinaryHeap<QueueEntry>,
    next_sequence: u64,
    queued_memory_bytes: u64,
}

impl PartitionScheduler {
    /// Construct a partition scheduler.
    pub(crate) fn new(id: PartitionSchedulerId, budget: SchedulerBudget) -> Self {
        Self {
            id,
            budget,
            pipelines: Vec::new(),
            queue: BinaryHeap::new(),
            next_sequence: 0,
            queued_memory_bytes: 0,
        }
    }

    /// Scheduler id.
    pub(crate) fn id(&self) -> PartitionSchedulerId {
        self.id
    }

    /// Number of queued events.
    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }

    /// Queued morsel memory.
    pub(crate) fn queued_memory_bytes(&self) -> u64 {
        self.queued_memory_bytes
    }

    /// Number of closed pipelines owned by this scheduler.
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Close a lowered pipeline with its source and return the
    /// scheduler-local opaque id.
    pub(crate) fn close_pipeline_with_source(
        &mut self,
        source: SchedulerPipelineSource,
    ) -> PipelineId {
        let id = PipelineId::new(self.pipelines.len());
        self.pipelines.push(SchedulerPipelineState::new(source));
        id
    }

    fn pipeline_source(&self, pipeline: PipelineId) -> Option<&SchedulerPipelineSource> {
        self.pipelines
            .get(pipeline.index())
            .map(|state| &state.source)
    }

    fn pop_task(&mut self, now_tick: u64) -> Option<SchedulerTask> {
        self.refresh_priorities(now_tick);
        let entry = self.queue.pop()?;
        self.queued_memory_bytes = self
            .queued_memory_bytes
            .saturating_sub(entry.task.memory_bytes());
        Some(entry.task)
    }

    /// Enqueue a task. Returns `false` when the bounded queue would
    /// exceed its task or memory budget.
    pub(crate) fn enqueue(&mut self, task: SchedulerTask, now_tick: u64) -> bool {
        let memory_bytes = task.memory_bytes();
        if self.queue.len() >= self.budget.max_queued_events
            || self.queued_memory_bytes.saturating_add(memory_bytes)
                > self.budget.max_queued_memory_bytes
        {
            return false;
        }

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        let priority = task.priority(now_tick, now_tick);
        self.queued_memory_bytes = self.queued_memory_bytes.saturating_add(memory_bytes);
        self.queue.push(QueueEntry {
            priority,
            sequence,
            ready_tick: now_tick,
            task,
        });
        true
    }

    /// Recompute priority keys, mainly to age long-waiting morsels and
    /// incorporate changed scheduler state after balancing.
    pub(crate) fn refresh_priorities(&mut self, now_tick: u64) {
        let entries = std::mem::take(&mut self.queue);
        self.queue = entries
            .into_iter()
            .map(|mut entry| {
                entry.priority = entry.task.priority(entry.ready_tick, now_tick);
                entry
            })
            .collect();
    }

    /// Advance one task.
    pub(crate) fn make_progress(&mut self, now_tick: u64) -> Option<SchedulerStep> {
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        self.poll_progress(now_tick, &mut cx)
    }

    /// Poll one scheduler task.
    pub(crate) fn poll_progress(
        &mut self,
        now_tick: u64,
        cx: &mut Context<'_>,
    ) -> Option<SchedulerStep> {
        match self.pop_task(now_tick)? {
            SchedulerTask::Work(mut work) => {
                let morsel_id = work.morsel.id;
                let pipeline = work.pipeline;
                if let Some((from_stage, to_stage)) = work.morsel.advance_one_stage() {
                    let _requeued = self.enqueue(SchedulerTask::Work(work), now_tick);
                    Some(SchedulerStep::Advanced {
                        morsel_id,
                        pipeline,
                        from_stage,
                        to_stage,
                    })
                } else {
                    Some(SchedulerStep::Completed {
                        morsel_id,
                        pipeline,
                    })
                }
            }
            SchedulerTask::Segment(mut request) => match request.poll(cx) {
                Poll::Ready(Ok(bytes)) => Some(SchedulerStep::CompletedSegment {
                    request_id: request.id,
                    pipeline: request.pipeline,
                    segment_id: request.segment_id,
                    bytes,
                }),
                Poll::Ready(Err(_err)) => Some(SchedulerStep::CompletedSegment {
                    request_id: request.id,
                    pipeline: request.pipeline,
                    segment_id: request.segment_id,
                    bytes: request.bytes,
                }),
                Poll::Pending => {
                    let request_id = request.id;
                    let pipeline = request.pipeline;
                    let segment_id = request.segment_id;
                    let _requeued = self.enqueue(SchedulerTask::Segment(request), now_tick);
                    Some(SchedulerStep::PendingSegment {
                        request_id,
                        pipeline,
                        segment_id,
                    })
                }
            },
            SchedulerTask::Control(event) => Some(SchedulerStep::Control { event }),
        }
    }

    /// Drain up to `max_morsels` lower-priority data morsels for a
    /// future work-stealing implementation.
    pub(crate) fn stealable_morsels(&mut self, max_morsels: usize) -> Vec<SchedulerMorsel> {
        let mut kept = BinaryHeap::new();
        let mut stolen = Vec::new();
        while let Some(entry) = self.queue.pop() {
            let QueueEntry {
                priority,
                sequence,
                ready_tick,
                task,
            } = entry;
            match task {
                SchedulerTask::Work(work)
                    if stolen.len() < max_morsels
                        && !matches!(
                            work.morsel.role,
                            MorselRole::Sink | MorselRole::InformationProducer
                        ) =>
                {
                    self.queued_memory_bytes =
                        self.queued_memory_bytes.saturating_sub(work.memory_bytes());
                    stolen.push(work.morsel);
                }
                task => kept.push(QueueEntry {
                    priority,
                    sequence,
                    ready_tick,
                    task,
                }),
            }
        }
        self.queue = kept;
        stolen
    }
}

/// A lowered layout-plan node recorded by the scheduler prototype.
///
/// This is intentionally descriptive metadata, not an executable plan
/// node. The executable unit in this prototype is the scheduler event
/// registered by leaves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweredLayoutNode {
    id: SubplanId,
    local_range: Range<u64>,
    global_range: Range<u64>,
    schema: String,
    child_count: usize,
}

impl LoweredLayoutNode {
    pub(crate) fn id(&self) -> SubplanId {
        self.id
    }

    pub(crate) fn child_count(&self) -> usize {
        self.child_count
    }

    pub(crate) fn global_range(&self) -> &Range<u64> {
        &self.global_range
    }
}

/// One initial leaf work item produced by layout lowering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweredLeafWork {
    subplan: SubplanId,
    pipeline: PipelineId,
    morsel: MorselId,
    local_range: Range<u64>,
    global_range: Range<u64>,
    role: MorselRole,
    schema: String,
}

impl LoweredLeafWork {
    pub(crate) fn pipeline(&self) -> PipelineId {
        self.pipeline
    }

    pub(crate) fn morsel(&self) -> MorselId {
        self.morsel
    }

    pub(crate) fn local_range(&self) -> &Range<u64> {
        &self.local_range
    }

    pub(crate) fn role(&self) -> MorselRole {
        self.role
    }

    pub(crate) fn global_range(&self) -> &Range<u64> {
        &self.global_range
    }
}

/// Summary returned by driving the single-scheduler prototype.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LayoutSchedulerRunReport {
    steps: usize,
    advanced_morsels: usize,
    completed_morsels: usize,
    completed_segments: usize,
    pending_segments: usize,
    control_events: usize,
}

impl LayoutSchedulerRunReport {
    /// Number of scheduler steps executed.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Number of morsels completed.
    pub fn completed_morsels(&self) -> usize {
        self.completed_morsels
    }

    /// Number of segment futures completed.
    pub fn completed_segments(&self) -> usize {
        self.completed_segments
    }

    /// Number of segment futures that were polled but not ready.
    pub fn pending_segments(&self) -> usize {
        self.pending_segments
    }
}

/// Lowering context for the single-scheduler layout prototype.
///
/// This is the bridge between the recursive [`crate::v2::plan::LayoutPlan`]
/// tree and the scheduler sketch above. Plan nodes record metadata;
/// leaves enqueue initial morsels into one partition-local scheduler.
/// The scheduler can then be driven by repeatedly popping the highest
/// priority event and executing one abstract stage.
pub struct LayoutLoweringCtx {
    scheduler: PartitionScheduler,
    domain: DomainId,
    current_global_range: Range<u64>,
    next_subplan: u32,
    next_morsel: u64,
    next_io_request: u64,
    nodes: Vec<LoweredLayoutNode>,
    leaves: Vec<LoweredLeafWork>,
}

impl LayoutLoweringCtx {
    /// Construct a lowering context for one scheduler over one ordinal
    /// row domain.
    pub fn for_single_scheduler(total_rows: u64) -> Self {
        Self::with_budget(total_rows, SchedulerBudget::default())
    }

    pub(crate) fn with_budget(total_rows: u64, budget: SchedulerBudget) -> Self {
        Self {
            scheduler: PartitionScheduler::new(PartitionSchedulerId::new(0), budget),
            domain: DomainId::new(0),
            current_global_range: 0..total_rows,
            next_subplan: 1,
            next_morsel: 1,
            next_io_request: 1,
            nodes: Vec::new(),
            leaves: Vec::new(),
        }
    }

    /// Run a lowering step while mapping the callee's local
    /// coordinates to `global_range` in the root scheduler domain.
    pub(crate) fn with_global_range<R>(
        &mut self,
        global_range: Range<u64>,
        f: impl FnOnce(&mut Self) -> VortexResult<R>,
    ) -> VortexResult<R> {
        let previous = std::mem::replace(&mut self.current_global_range, global_range);
        let result = f(self);
        self.current_global_range = previous;
        result
    }

    pub(crate) fn current_global_range(&self) -> Range<u64> {
        self.current_global_range.clone()
    }

    /// Record a plan node and return its prototype sub-plan id.
    pub(crate) fn register_plan_node(
        &mut self,
        local_range: Range<u64>,
        schema: &DType,
        child_count: usize,
    ) -> SubplanId {
        let id = self.alloc_subplan();
        self.nodes.push(LoweredLayoutNode {
            id,
            local_range,
            global_range: self.current_global_range.clone(),
            schema: schema.to_string(),
            child_count,
        });
        id
    }

    /// Register initial work for a leaf in the current global range.
    pub(crate) fn register_leaf_work(
        &mut self,
        subplan: SubplanId,
        local_range: Range<u64>,
        schema: &DType,
    ) -> VortexResult<()> {
        let role = role_for_schema(schema);
        let global_range = self.current_global_range.clone();
        let pipeline = self.close_pipeline_with_source(SchedulerPipelineSource::Leaf {
            subplan,
            local_range: local_range.clone(),
            global_range: global_range.clone(),
            schema: schema.to_string(),
            role,
        });
        let morsel_id = self.alloc_morsel();
        let estimate = estimate_for_leaf(&global_range, schema);
        let priority = priority_for_leaf(role, &global_range, estimate);
        let stage_count = stage_count_for_role(role);
        let morsel = SchedulerMorsel::new(
            morsel_id,
            self.domain,
            global_range.clone(),
            role,
            stage_count,
            estimate,
            priority,
        );
        let work = SchedulerWorkTask::new(pipeline, morsel);

        if !self.scheduler.enqueue(SchedulerTask::Work(work), 0) {
            vortex_bail!(
                "layout scheduler queue full while registering leaf work for {global_range:?}"
            );
        }

        self.leaves.push(LoweredLeafWork {
            subplan,
            pipeline,
            morsel: morsel_id,
            local_range,
            global_range,
            role,
            schema: schema.to_string(),
        });
        Ok(())
    }

    /// Close a pipeline with an abstract leaf source.
    pub(crate) fn close_pipeline_with_source(
        &mut self,
        source: SchedulerPipelineSource,
    ) -> PipelineId {
        self.scheduler.close_pipeline_with_source(source)
    }

    /// Close a pipeline with a segment source and return its
    /// scheduler-local pipeline id.
    pub(crate) fn close_pipeline_with_segment_source(
        &mut self,
        subplan: SubplanId,
        segment_id: SegmentId,
        local_range: Range<u64>,
        schema: &DType,
    ) -> PipelineId {
        self.close_pipeline_with_source(SchedulerPipelineSource::Segment {
            subplan,
            segment_id,
            local_range,
            global_range: self.current_global_range.clone(),
            schema: schema.to_string(),
        })
    }

    /// Close a pipeline whose source runs an already-built plan into
    /// the scheduler sink. This is the first runnable bridge; finer
    /// lowering can replace it one plan node at a time.
    pub(crate) fn close_pipeline_with_execute_source(
        &mut self,
        plan: LayoutPlanRef,
        row_range: Range<u64>,
        demand: RowDemand,
        frontier: OutputFrontier,
        ctx: ScanCtx,
    ) -> PipelineId {
        self.close_pipeline_with_source(SchedulerPipelineSource::ExecutePlan {
            plan,
            row_range,
            demand,
            frontier,
            ctx,
        })
    }

    /// Enqueue work for an already-closed pipeline.
    pub(crate) fn enqueue_pipeline_work(
        &mut self,
        pipeline: PipelineId,
        global_range: Range<u64>,
        schema: &DType,
    ) -> VortexResult<MorselId> {
        if pipeline.index() >= self.scheduler.pipeline_count() {
            vortex_bail!(
                "work task referenced unknown pipeline index {}",
                pipeline.index()
            );
        }
        let role = role_for_schema(schema);
        let morsel_id = self.alloc_morsel();
        let estimate = estimate_for_leaf(&global_range, schema);
        let priority = priority_for_leaf(role, &global_range, estimate);
        let stage_count = 1;
        let morsel = SchedulerMorsel::new(
            morsel_id,
            self.domain,
            global_range,
            role,
            stage_count,
            estimate,
            priority,
        );
        let work = SchedulerWorkTask::new(pipeline, morsel);
        if !self.scheduler.enqueue(SchedulerTask::Work(work), 0) {
            vortex_bail!(
                "layout scheduler queue full while registering pipeline work for {pipeline:?}"
            );
        }
        Ok(morsel_id)
    }

    /// Register a segment future with the scheduler.
    pub(crate) fn register_segment_task(
        &mut self,
        pipeline: PipelineId,
        segment_id: SegmentId,
        range: Range<u64>,
        bytes: u64,
        segment_future: SharedSegmentFuture,
    ) -> VortexResult<IoRequestId> {
        if pipeline.index() >= self.scheduler.pipeline_count() {
            vortex_bail!(
                "segment task referenced unknown pipeline index {}",
                pipeline.index()
            );
        }
        let request_id = self.alloc_io_request();
        let rows = range.end.saturating_sub(range.start).max(1);
        let priority = MorselPriority::value_work(rows.saturating_mul(10), bytes.max(1), 0);
        let task = SchedulerSegmentTask::new(
            request_id,
            pipeline,
            segment_id,
            self.domain,
            range.clone(),
            bytes,
            priority,
            segment_future,
        );
        if !self.scheduler.enqueue(SchedulerTask::Segment(task), 0) {
            vortex_bail!(
                "layout scheduler queue full while registering segment work for {range:?}"
            );
        }
        Ok(request_id)
    }

    /// Number of lowered plan nodes.
    pub fn lowered_node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of initial leaf work items.
    pub fn leaf_work_count(&self) -> usize {
        self.leaves.len()
    }

    /// Number of queued scheduler events.
    pub fn queued_event_count(&self) -> usize {
        self.scheduler.len()
    }

    /// Number of closed pipelines.
    pub fn pipeline_count(&self) -> usize {
        self.scheduler.pipeline_count()
    }

    /// Queued scheduler memory estimate.
    pub fn queued_memory_bytes(&self) -> u64 {
        self.scheduler.queued_memory_bytes()
    }

    pub(crate) fn lowered_nodes(&self) -> &[LoweredLayoutNode] {
        &self.nodes
    }

    pub(crate) fn leaf_work(&self) -> &[LoweredLeafWork] {
        &self.leaves
    }

    /// Drive the scheduler until no events remain.
    pub fn drive_to_completion(&mut self) -> LayoutSchedulerRunReport {
        let mut report = LayoutSchedulerRunReport::default();
        let mut tick = 0;
        while let Some(step) = self.scheduler.make_progress(tick) {
            report.steps += 1;
            match step {
                SchedulerStep::Advanced { .. } => report.advanced_morsels += 1,
                SchedulerStep::Completed { .. } => report.completed_morsels += 1,
                SchedulerStep::CompletedSegment { .. } => report.completed_segments += 1,
                SchedulerStep::PendingSegment { .. } => {
                    report.pending_segments += 1;
                    break;
                }
                SchedulerStep::Control { .. } => report.control_events += 1,
            }
            tick = tick.saturating_add(1);
        }
        report
    }

    pub(crate) fn drain_steps(&mut self) -> Vec<SchedulerStep> {
        let mut steps = Vec::new();
        let mut tick = 0;
        while let Some(step) = self.scheduler.make_progress(tick) {
            steps.push(step);
            tick = tick.saturating_add(1);
        }
        steps
    }

    async fn drive_to_sink(
        mut self,
        sink: kanal::AsyncSender<VortexResult<ArrayRef>>,
    ) -> VortexResult<LayoutSchedulerRunReport> {
        let mut report = LayoutSchedulerRunReport::default();
        let mut tick = 0;
        while let Some(task) = self.scheduler.pop_task(tick) {
            report.steps += 1;
            match task {
                SchedulerTask::Work(mut work) => {
                    match self.scheduler.pipeline_source(work.pipeline).cloned() {
                        Some(SchedulerPipelineSource::ExecutePlan {
                            plan,
                            row_range,
                            demand,
                            frontier,
                            ctx,
                        }) => {
                            report.completed_morsels += 1;
                            let mut stream = plan.execute(row_range, &demand, &frontier, &ctx)?;
                            while let Some(array) = stream.next().await {
                                if sink.send(array).await.is_err() {
                                    return Ok(report);
                                }
                            }
                        }
                        _ => {
                            let morsel_id = work.morsel.id;
                            let pipeline = work.pipeline;
                            if work.morsel.advance_one_stage().is_some() {
                                if !self.scheduler.enqueue(SchedulerTask::Work(work), tick) {
                                    vortex_bail!(
                                        "layout scheduler queue full while requeueing work for {pipeline:?}"
                                    );
                                }
                                report.advanced_morsels += 1;
                            } else {
                                let _ = morsel_id;
                                report.completed_morsels += 1;
                            }
                        }
                    }
                }
                SchedulerTask::Segment(mut segment) => {
                    let _bytes = segment.wait().await?;
                    report.completed_segments += 1;
                    // The next step is to store the segment bytes in
                    // this pipeline's local state and enqueue pipeline
                    // work that decodes and pushes to the sink.
                }
                SchedulerTask::Control(_) => report.control_events += 1,
            }
            tick = tick.saturating_add(1);
        }
        Ok(report)
    }

    fn alloc_subplan(&mut self) -> SubplanId {
        let id = SubplanId::new(self.next_subplan);
        self.next_subplan = self.next_subplan.saturating_add(1);
        id
    }

    fn alloc_morsel(&mut self) -> MorselId {
        let id = MorselId::new(self.next_morsel);
        self.next_morsel = self.next_morsel.saturating_add(1);
        id
    }

    fn alloc_io_request(&mut self) -> IoRequestId {
        let id = IoRequestId::new(self.next_io_request);
        self.next_io_request = self.next_io_request.saturating_add(1);
        id
    }
}

/// Execute one partition by spawning a scheduler driver and returning
/// a stream over its sink queue.
///
/// This is intentionally a compatibility bridge: the root pipeline
/// source delegates to the existing `LayoutPlan::execute` so the
/// scheduler/queue shape can run end-to-end before every plan node has
/// a native pipeline implementation.
pub(crate) fn execute_with_single_scheduler(
    plan: LayoutPlanRef,
    row_range: Range<u64>,
    demand: RowDemand,
    frontier: OutputFrontier,
    ctx: ScanCtx,
) -> VortexResult<SendableArrayStream> {
    let dtype = plan.schema().clone();
    let mut lowering = LayoutLoweringCtx::for_single_scheduler(row_range.end);
    let pipeline = lowering.close_pipeline_with_execute_source(
        Arc::clone(&plan),
        row_range.clone(),
        demand,
        frontier,
        ctx.clone(),
    );
    lowering.enqueue_pipeline_work(pipeline, row_range, &dtype)?;

    let (sink_tx, sink_rx) = kanal::bounded_async::<VortexResult<ArrayRef>>(2);
    let driver_tx = sink_tx.clone();
    ctx.session()
        .handle()
        .spawn(async move {
            if let Err(err) = lowering.drive_to_sink(driver_tx.clone()).await {
                drop(driver_tx.send(Err(err)).await);
            }
        })
        .detach();

    let stream = try_stream! {
        while let Ok(item) = sink_rx.recv().await {
            yield item?;
        }
    };
    Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
}

fn role_for_schema(schema: &DType) -> MorselRole {
    if matches!(schema, DType::Bool(_)) {
        MorselRole::InformationProducer
    } else {
        MorselRole::ValueProducer
    }
}

fn estimate_for_leaf(range: &Range<u64>, schema: &DType) -> MorselEstimate {
    let rows = range.end.saturating_sub(range.start).max(1);
    let bytes_per_row = match schema {
        DType::Bool(_) => 1,
        DType::Primitive(..) => 8,
        DType::Utf8(_) | DType::Binary(_) => 32,
        DType::Struct(..) => 64,
        _ => 16,
    };
    let io_bytes = rows.saturating_mul(bytes_per_row);
    MorselEstimate::new(rows.saturating_mul(10), io_bytes, io_bytes.min(1024 * 1024))
}

fn priority_for_leaf(
    role: MorselRole,
    range: &Range<u64>,
    estimate: MorselEstimate,
) -> MorselPriority {
    let rows = range.end.saturating_sub(range.start).max(1);
    match role {
        MorselRole::InformationProducer => MorselPriority::information_producer(
            rows.saturating_mul(100),
            estimate.cpu_ns.saturating_add(estimate.io_bytes).max(1),
            10_000,
        ),
        _ => MorselPriority::value_work(
            rows.saturating_mul(10),
            estimate.cpu_ns.saturating_add(estimate.io_bytes).max(1),
            0,
        ),
    }
}

fn stage_count_for_role(role: MorselRole) -> u16 {
    match role {
        MorselRole::InformationProducer => 3,
        MorselRole::InformationConsumer
        | MorselRole::ValueProducer
        | MorselRole::Combiner
        | MorselRole::Sink => 2,
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hash;
    use std::sync::Arc;

    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::stream::SendableArrayStream;

    use super::*;
    use crate::v2::chunked::ChunkedPlan;
    use crate::v2::demand::RowDemand;
    use crate::v2::plan::LayoutPlan;
    use crate::v2::plan::LayoutPlanRef;
    use crate::v2::plan::PartitionStats;
    use crate::v2::plan::lower_to_single_scheduler;
    use crate::v2::scan_ctx::ScanCtx;

    const ROWS: DomainId = DomainId::new(0);
    const CHEAP_FILTER: SubplanId = SubplanId::new(1);
    const EXPENSIVE_FILTER: SubplanId = SubplanId::new(2);
    const THIRD_FILTER: SubplanId = SubplanId::new(3);

    fn high_value_estimate() -> WorkEstimate {
        WorkEstimate::new(1.0, 100.0, 0.95, 0.9)
    }

    fn low_value_estimate() -> WorkEstimate {
        WorkEstimate::new(100.0, 1.0, 0.05, 0.5)
    }

    fn narrow_output(rows: u64) -> OutputEstimate {
        OutputEstimate::new(rows, rows * 8)
    }

    fn i64_dtype() -> DType {
        DType::Primitive(PType::I64, Nullability::NonNullable)
    }

    fn bool_dtype() -> DType {
        DType::Bool(Nullability::NonNullable)
    }

    struct TestLeaf {
        tag: &'static str,
        dtype: DType,
        row_count: u64,
    }

    impl TestLeaf {
        fn new(tag: &'static str, dtype: DType, row_count: u64) -> Self {
            Self {
                tag,
                dtype,
                row_count,
            }
        }
    }

    impl PartialEq for TestLeaf {
        fn eq(&self, other: &Self) -> bool {
            self.tag == other.tag && self.dtype == other.dtype && self.row_count == other.row_count
        }
    }

    impl Eq for TestLeaf {}

    impl Hash for TestLeaf {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.tag.hash(state);
            self.dtype.hash(state);
            self.row_count.hash(state);
        }
    }

    impl LayoutPlan for TestLeaf {
        fn schema(&self) -> &DType {
            &self.dtype
        }

        fn partition_count(&self) -> usize {
            1
        }

        fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
            if partition != 0 {
                vortex_bail!("TestLeaf partition out of range");
            }
            Ok(PartitionStats::for_range(0..self.row_count))
        }

        fn output_ordered(&self) -> bool {
            true
        }

        fn required_input_ordered(&self) -> Vec<bool> {
            vec![]
        }

        fn maintains_input_order(&self) -> Vec<bool> {
            vec![]
        }

        fn children(&self) -> &[LayoutPlanRef] {
            &[]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<LayoutPlanRef>,
        ) -> VortexResult<LayoutPlanRef> {
            if !children.is_empty() {
                vortex_bail!("TestLeaf has no children");
            }
            Ok(self)
        }

        fn execute(
            &self,
            _row_range: Range<u64>,
            _demand: &RowDemand,
            _frontier: &OutputFrontier,
            _ctx: &ScanCtx,
        ) -> VortexResult<SendableArrayStream> {
            unreachable!("TestLeaf is lowering-only")
        }
    }

    struct TestContainer {
        children: Vec<LayoutPlanRef>,
        dtype: DType,
        row_count: u64,
    }

    impl TestContainer {
        fn new(children: Vec<LayoutPlanRef>, row_count: u64) -> Self {
            Self {
                children,
                dtype: i64_dtype(),
                row_count,
            }
        }
    }

    impl PartialEq for TestContainer {
        fn eq(&self, other: &Self) -> bool {
            crate::v2::plan::plan_slices_eq(&self.children, &other.children)
                && self.row_count == other.row_count
        }
    }

    impl Eq for TestContainer {}

    impl Hash for TestContainer {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            crate::v2::plan::hash_plan_slice(&self.children, state);
            self.row_count.hash(state);
        }
    }

    impl LayoutPlan for TestContainer {
        fn schema(&self) -> &DType {
            &self.dtype
        }

        fn partition_count(&self) -> usize {
            1
        }

        fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
            if partition != 0 {
                vortex_bail!("TestContainer partition out of range");
            }
            Ok(PartitionStats::for_range(0..self.row_count))
        }

        fn output_ordered(&self) -> bool {
            true
        }

        fn required_input_ordered(&self) -> Vec<bool> {
            vec![true; self.children.len()]
        }

        fn maintains_input_order(&self) -> Vec<bool> {
            vec![true; self.children.len()]
        }

        fn children(&self) -> &[LayoutPlanRef] {
            &self.children
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<LayoutPlanRef>,
        ) -> VortexResult<LayoutPlanRef> {
            Ok(Arc::new(Self {
                children,
                dtype: self.dtype.clone(),
                row_count: self.row_count,
            }))
        }

        fn execute(
            &self,
            _row_range: Range<u64>,
            _demand: &RowDemand,
            _frontier: &OutputFrontier,
            _ctx: &ScanCtx,
        ) -> VortexResult<SendableArrayStream> {
            unreachable!("TestContainer is lowering-only")
        }
    }

    #[test]
    fn unknown_demand_is_correct_but_waitable() -> VortexResult<()> {
        let demand = OrdinalDemand::new(ROWS, 1_000);
        let target = 100..200;

        assert_eq!(demand.coverage(&target)?, Coverage::Unknown);
        let correctness_mask = demand.mask_for(&target)?;
        assert!(correctness_mask.all_true());

        let policy = PermitPolicy::new(128, 16, 1.0);
        let permit = policy.value_consumer_permit(&demand, &target, high_value_estimate())?;
        assert_eq!(permit.reason(), PermitReason::WaitForDemand);
        assert_eq!(permit.rows_to_poll(), 0);
        Ok(())
    }

    #[test]
    fn demand_producer_runs_to_first_uncovered_frontier() -> VortexResult<()> {
        let mut demand = OrdinalDemand::new(ROWS, 1_000);
        let policy = PermitPolicy::new(128, 16, 1.0);
        let target = 0..512;

        let permit = policy.demand_producer_permit(&demand, &target)?;
        assert_eq!(permit.reason(), PermitReason::DriveDemandProducer);
        assert_eq!(permit.range(), &(0..128));

        demand.publish(0..128, Mask::new_false(128))?;
        let permit = policy.demand_producer_permit(&demand, &target)?;
        assert_eq!(permit.range(), &(128..256));
        Ok(())
    }

    #[test]
    fn all_false_covered_prefix_skips_value_work() -> VortexResult<()> {
        let mut demand = OrdinalDemand::new(ROWS, 1_000);
        demand.publish(0..128, Mask::new_false(128))?;

        let policy = PermitPolicy::new(128, 16, 1.0);
        let permit = policy.value_consumer_permit(&demand, &(0..512), high_value_estimate())?;

        assert_eq!(permit.reason(), PermitReason::SkipAllFalse);
        assert_eq!(permit.range(), &(0..128));
        assert_eq!(permit.rows_to_poll(), 0);
        Ok(())
    }

    #[test]
    fn known_live_prefix_allows_value_work() -> VortexResult<()> {
        let mut demand = OrdinalDemand::new(ROWS, 1_000);
        demand.publish(0..128, Mask::new_true(128))?;

        let policy = PermitPolicy::new(128, 16, 1.0);
        let permit = policy.value_consumer_permit(&demand, &(0..512), high_value_estimate())?;

        assert_eq!(permit.reason(), PermitReason::ProceedWithKnownDemand);
        assert_eq!(permit.range(), &(0..128));
        assert_eq!(permit.rows_to_poll(), 128);
        Ok(())
    }

    #[test]
    fn low_value_unknown_range_gets_small_speculative_permit() -> VortexResult<()> {
        let demand = OrdinalDemand::new(ROWS, 1_000);
        let policy = PermitPolicy::new(128, 16, 1.0);
        let permit = policy.value_consumer_permit(&demand, &(0..512), low_value_estimate())?;

        assert_eq!(permit.reason(), PermitReason::Speculate);
        assert_eq!(permit.range(), &(0..16));
        assert_eq!(permit.rows_to_poll(), 16);
        Ok(())
    }

    #[test]
    fn sorted_domain_can_advertise_ordinal_lowering() {
        let sorted = Domain::Sorted {
            id: DomainId::new(1),
            key: "event_time",
            ordinal: ROWS,
        };
        let keyed = Domain::Keyed {
            id: DomainId::new(2),
            key: "user_id",
        };

        assert_eq!(sorted.ordinal_mapping(), Some(ROWS));
        assert_eq!(keyed.ordinal_mapping(), None);
    }

    #[test]
    fn output_grants_are_scoped_by_domain_and_subplan() -> VortexResult<()> {
        let cheap_key = GrantKey::new(ROWS, CHEAP_FILTER);
        let expensive_key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
        let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
        grantor.register_domain(ROWS, 1_000_000);
        grantor.set_frontier(cheap_key, 128 * 1024)?;
        grantor.set_frontier(expensive_key, 8 * 1024)?;

        let cheap_grant = grantor.grant(OutputGrantRequest::new(
            cheap_key,
            0..1_000_000,
            OutputEstimate::new(1_000_000, 8_000_000),
        ))?;
        let expensive_grant = grantor.grant(OutputGrantRequest::new(
            expensive_key,
            0..1_000_000,
            OutputEstimate::new(1_000_000, 8_000_000),
        ))?;

        assert_eq!(cheap_grant.reason(), OutputGrantReason::Granted);
        assert_eq!(cheap_grant.range(), &(0..64 * 1024));
        assert_eq!(cheap_grant.visible_frontier(), 128 * 1024);
        assert_eq!(expensive_grant.reason(), OutputGrantReason::Granted);
        assert_eq!(expensive_grant.range(), &(0..8 * 1024));
        assert_eq!(expensive_grant.visible_frontier(), 8 * 1024);
        Ok(())
    }

    #[test]
    fn output_grant_blocks_at_visible_frontier() -> VortexResult<()> {
        let key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
        let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
        grantor.register_domain(ROWS, 1_000_000);
        grantor.set_frontier(key, 8 * 1024)?;

        let grant = grantor.grant(OutputGrantRequest::new(
            key,
            8 * 1024..1_000_000,
            OutputEstimate::new(1_000_000, 8_000_000),
        ))?;

        assert_eq!(grant.reason(), OutputGrantReason::BlockedAtFrontier);
        assert_eq!(grant.range(), &(8 * 1024..8 * 1024));
        Ok(())
    }

    #[test]
    fn output_grant_uses_byte_cap_for_wide_rows() -> VortexResult<()> {
        let key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
        let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
        grantor.register_domain(ROWS, 1_000_000);
        grantor.set_frontier(key, 1_000_000)?;

        let grant = grantor.grant(OutputGrantRequest::new(
            key,
            0..1_000_000,
            OutputEstimate::new(1_000_000, 256_000_000),
        ))?;

        assert_eq!(grant.reason(), OutputGrantReason::Granted);
        assert_eq!(grant.range(), &(0..4096));
        assert_eq!(grant.estimate().rows(), 4096);
        assert_eq!(grant.estimate().bytes(), 1024 * 1024);
        Ok(())
    }

    #[test]
    fn output_grant_frontier_can_advance_after_demand_publication() -> VortexResult<()> {
        let key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
        let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
        grantor.register_domain(ROWS, 1_000_000);
        grantor.set_frontier(key, 0)?;

        let blocked = grantor.grant(OutputGrantRequest::new(
            key,
            0..1_000_000,
            OutputEstimate::new(1_000_000, 8_000_000),
        ))?;
        assert_eq!(blocked.reason(), OutputGrantReason::BlockedAtFrontier);

        grantor.advance_frontier(key, 32 * 1024)?;
        let grant = grantor.grant(OutputGrantRequest::new(
            key,
            0..1_000_000,
            OutputEstimate::new(1_000_000, 8_000_000),
        ))?;

        assert_eq!(grant.reason(), OutputGrantReason::Granted);
        assert_eq!(grant.range(), &(0..32 * 1024));
        Ok(())
    }

    #[test]
    fn output_frontier_grants_sequentially() -> VortexResult<()> {
        let mut frontier = OutputFrontier::unbounded(100).clone_with_offset(10..30);

        let first = frontier.grant_next(8, narrow_output(20))?;
        let second = frontier.grant_next(20, narrow_output(20))?;

        assert_eq!(first.range(), &(0..8));
        assert_eq!(second.range(), &(8..20));
        Ok(())
    }

    #[test]
    fn output_frontier_sideways_clones_have_independent_cursors() -> VortexResult<()> {
        let frontier = OutputFrontier::unbounded(100).clone_with_offset(10..50);
        let mut cheap = frontier.clone_sideways(CHEAP_FILTER);
        let mut expensive = frontier.clone_sideways(EXPENSIVE_FILTER);

        let cheap_grant = cheap.grant_next(16, narrow_output(40))?;
        let expensive_grant = expensive.grant_next(4, narrow_output(40))?;

        assert_eq!(cheap_grant.range(), &(0..16));
        assert_eq!(expensive_grant.range(), &(0..4));
        Ok(())
    }

    #[test]
    fn output_frontier_offset_clone_maps_grants_back_to_local_rows() -> VortexResult<()> {
        let key = GrantKey::new(ROWS, CHEAP_FILTER);
        let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
        grantor.register_domain(ROWS, 100);
        grantor.set_frontier(key, 35)?;
        let source: Arc<dyn FrontierSource> = Arc::new(parking_lot::Mutex::new(grantor));
        let mut frontier = OutputFrontier::new(source, key, 100).clone_with_offset(20..60);

        let grant = frontier.grant_next(40, narrow_output(40))?;

        assert_eq!(grant.reason(), OutputGrantReason::Granted);
        assert_eq!(grant.range(), &(0..15));
        assert_eq!(grant.visible_frontier(), 15);
        Ok(())
    }

    #[test]
    fn conjunct_controller_releases_distinct_initial_frontiers() -> VortexResult<()> {
        let policy = ConjunctFrontierPolicy::new(128 * 1024, 8 * 1024, 64 * 1024, 1024 * 1024);
        let mut controller = ConjunctFrontierController::new(
            ROWS,
            1_000_000,
            vec![CHEAP_FILTER, EXPENSIVE_FILTER],
            policy,
        )?;

        controller.begin_range(&(0..1_000_000))?;

        assert_eq!(controller.stage_frontier(0)?, 128 * 1024);
        assert_eq!(controller.stage_frontier(1)?, 8 * 1024);

        let leader = controller.grant_for_stage(0, 0..1_000_000, narrow_output(1_000_000))?;
        let dependent = controller.grant_for_stage(1, 0..1_000_000, narrow_output(1_000_000))?;

        assert_eq!(leader.range(), &(0..64 * 1024));
        assert_eq!(dependent.range(), &(0..8 * 1024));
        Ok(())
    }

    #[test]
    fn conjunct_controller_releases_next_stage_to_known_demand_prefix() -> VortexResult<()> {
        let policy = ConjunctFrontierPolicy::new(128 * 1024, 0, 128 * 1024, 1024 * 1024);
        let mut controller = ConjunctFrontierController::new(
            ROWS,
            1_000_000,
            vec![CHEAP_FILTER, EXPENSIVE_FILTER],
            policy,
        )?;
        controller.begin_range(&(0..1_000_000))?;

        let blocked = controller.grant_for_stage(1, 0..1_000_000, narrow_output(1_000_000))?;
        assert_eq!(blocked.reason(), OutputGrantReason::BlockedAtFrontier);

        let mut demand = OrdinalDemand::new(ROWS, 1_000_000);
        demand.publish(0..32 * 1024, Mask::new_true(32 * 1024))?;
        controller.release_after_stage(0, &demand, &(0..1_000_000))?;

        assert_eq!(controller.stage_frontier(1)?, 32 * 1024);
        let grant = controller.grant_for_stage(1, 0..1_000_000, narrow_output(1_000_000))?;
        assert_eq!(grant.reason(), OutputGrantReason::Granted);
        assert_eq!(grant.range(), &(0..32 * 1024));
        Ok(())
    }

    #[test]
    fn conjunct_controller_releases_stage_by_stage() -> VortexResult<()> {
        let policy = ConjunctFrontierPolicy::new(128 * 1024, 0, 128 * 1024, 1024 * 1024);
        let mut controller = ConjunctFrontierController::new(
            ROWS,
            1_000_000,
            vec![CHEAP_FILTER, EXPENSIVE_FILTER, THIRD_FILTER],
            policy,
        )?;
        controller.begin_range(&(0..1_000_000))?;

        let mut first_demand = OrdinalDemand::new(ROWS, 1_000_000);
        first_demand.publish(0..64 * 1024, Mask::new_true(64 * 1024))?;
        controller.release_after_stage(0, &first_demand, &(0..1_000_000))?;

        assert_eq!(controller.stage_frontier(1)?, 64 * 1024);
        assert_eq!(controller.stage_frontier(2)?, 0);

        let mut second_demand = OrdinalDemand::new(ROWS, 1_000_000);
        second_demand.publish(0..16 * 1024, Mask::new_true(16 * 1024))?;
        controller.release_after_stage(1, &second_demand, &(0..1_000_000))?;

        assert_eq!(controller.stage_frontier(2)?, 16 * 1024);
        Ok(())
    }

    #[test]
    fn conjunct_controller_allows_bounded_dependent_speculation() -> VortexResult<()> {
        let policy = ConjunctFrontierPolicy::new(128 * 1024, 4 * 1024, 128 * 1024, 1024 * 1024);
        let mut controller = ConjunctFrontierController::new(
            ROWS,
            1_000_000,
            vec![CHEAP_FILTER, EXPENSIVE_FILTER],
            policy,
        )?;
        controller.begin_range(&(0..1_000_000))?;

        let mut demand = OrdinalDemand::new(ROWS, 1_000_000);
        demand.publish(0..16 * 1024, Mask::new_true(16 * 1024))?;
        controller.release_after_stage(0, &demand, &(0..1_000_000))?;

        assert_eq!(controller.stage_frontier(1)?, 20 * 1024);
        let grant =
            controller.grant_for_stage(1, 16 * 1024..1_000_000, narrow_output(1_000_000))?;
        assert_eq!(grant.reason(), OutputGrantReason::Granted);
        assert_eq!(grant.range(), &(16 * 1024..20 * 1024));
        Ok(())
    }

    fn scheduler_morsel(
        id: u64,
        role: MorselRole,
        order_key: Range<u64>,
        stage_count: u16,
        priority: MorselPriority,
    ) -> SchedulerMorsel {
        SchedulerMorsel::new(
            MorselId::new(id),
            ROWS,
            order_key,
            role,
            stage_count,
            MorselEstimate::new(10_000, 0, 1024),
            priority,
        )
    }

    #[test]
    fn partition_scheduler_prioritizes_information_over_row_offset() {
        let mut scheduler =
            PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
        assert_eq!(scheduler.id(), PartitionSchedulerId::new(0));

        let early_value = scheduler_morsel(
            1,
            MorselRole::ValueProducer,
            0..1024,
            2,
            MorselPriority::value_work(1_000, 10_000, 0),
        );
        let later_information = scheduler_morsel(
            2,
            MorselRole::InformationProducer,
            64 * 1024..65 * 1024,
            2,
            MorselPriority::information_producer(100_000, 1_000, 10_000),
        );

        assert!(scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), early_value)),
            0
        ));
        assert!(scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(
                PipelineId::new(0),
                later_information
            )),
            0
        ));

        assert_eq!(
            scheduler.make_progress(1),
            Some(SchedulerStep::Advanced {
                morsel_id: MorselId::new(2),
                pipeline: PipelineId::new(0),
                from_stage: 0,
                to_stage: 1,
            })
        );
    }

    #[test]
    fn partition_scheduler_advances_one_pipeline_stage_per_step() {
        let mut scheduler =
            PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
        let morsel = scheduler_morsel(
            7,
            MorselRole::InformationConsumer,
            0..1024,
            2,
            MorselPriority::value_work(10_000, 10_000, 0),
        );

        assert!(scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), morsel)),
            0
        ));
        assert_eq!(
            scheduler.make_progress(0),
            Some(SchedulerStep::Advanced {
                morsel_id: MorselId::new(7),
                pipeline: PipelineId::new(0),
                from_stage: 0,
                to_stage: 1,
            })
        );
        assert_eq!(
            scheduler.make_progress(0),
            Some(SchedulerStep::Completed {
                morsel_id: MorselId::new(7),
                pipeline: PipelineId::new(0),
            })
        );
        assert_eq!(scheduler.make_progress(0), None);
    }

    #[test]
    fn partition_scheduler_bounds_queued_memory() {
        let mut scheduler =
            PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::new(8, 1500));
        let first = scheduler_morsel(
            1,
            MorselRole::ValueProducer,
            0..1024,
            1,
            MorselPriority::value_work(1, 1, 0),
        );
        let second = scheduler_morsel(
            2,
            MorselRole::ValueProducer,
            1024..2048,
            1,
            MorselPriority::value_work(1, 1, 0),
        );

        assert!(scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), first)),
            0
        ));
        assert!(!scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), second)),
            0
        ));
        assert_eq!(scheduler.len(), 1);
        assert_eq!(scheduler.queued_memory_bytes(), 1024);
    }

    #[test]
    fn partition_scheduler_queue_holds_io_and_control_events() {
        let mut scheduler =
            PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
        let segment = SchedulerSegmentTask::metadata_only(
            IoRequestId::new(11),
            PipelineId::new(4),
            SegmentId::from(7),
            ROWS,
            0..4096,
            32 * 1024,
            MorselPriority::information_producer(10_000, 1_000, 0),
        );
        let control = SchedulerControlEvent::Rebalance {
            reason: "test rebalance",
        };

        assert!(scheduler.enqueue(SchedulerTask::Control(control.clone()), 0));
        assert!(scheduler.enqueue(SchedulerTask::Segment(segment), 0));

        assert_eq!(
            scheduler.make_progress(0),
            Some(SchedulerStep::CompletedSegment {
                request_id: IoRequestId::new(11),
                pipeline: PipelineId::new(4),
                segment_id: SegmentId::from(7),
                bytes: 32 * 1024,
            })
        );
        assert_eq!(
            scheduler.make_progress(0),
            Some(SchedulerStep::Control { event: control })
        );
    }

    #[test]
    fn partition_scheduler_steals_only_non_critical_data_morsels() {
        let mut scheduler =
            PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
        let data = scheduler_morsel(
            1,
            MorselRole::ValueProducer,
            0..1024,
            1,
            MorselPriority::value_work(1, 1, 0),
        );
        let information = scheduler_morsel(
            2,
            MorselRole::InformationProducer,
            1024..2048,
            1,
            MorselPriority::information_producer(100, 1, 0),
        );
        let sink = scheduler_morsel(
            3,
            MorselRole::Sink,
            2048..3072,
            1,
            MorselPriority::value_work(1, 1, 0),
        );

        assert!(scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), data)),
            0
        ));
        assert!(scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), information)),
            0
        ));
        assert!(scheduler.enqueue(
            SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), sink)),
            0
        ));

        let stolen = scheduler.stealable_morsels(4);
        assert_eq!(stolen.len(), 1);
        assert_eq!(stolen[0].id(), MorselId::new(1));
        assert_eq!(scheduler.len(), 2);
    }

    #[test]
    fn layout_lowering_registers_leaf_work() -> VortexResult<()> {
        let leaf = TestLeaf::new("bool-leaf", bool_dtype(), 100);
        let mut ctx = lower_to_single_scheduler(&leaf, 0..100)?;

        assert_eq!(ctx.lowered_node_count(), 1);
        assert_eq!(ctx.pipeline_count(), 1);
        assert_eq!(ctx.leaf_work_count(), 1);
        assert_eq!(ctx.queued_event_count(), 1);
        assert_eq!(ctx.leaf_work()[0].role(), MorselRole::InformationProducer);
        assert_eq!(ctx.leaf_work()[0].pipeline().index(), 0);

        let report = ctx.drive_to_completion();
        assert_eq!(report.completed_morsels(), 1);
        assert_eq!(report.steps(), 3);
        Ok(())
    }

    #[test]
    fn layout_lowering_prioritizes_information_leaves() -> VortexResult<()> {
        let value: LayoutPlanRef = Arc::new(TestLeaf::new("value", i64_dtype(), 100));
        let info: LayoutPlanRef = Arc::new(TestLeaf::new("info", bool_dtype(), 100));
        let plan = TestContainer::new(vec![Arc::clone(&value), Arc::clone(&info)], 100);
        let mut ctx = lower_to_single_scheduler(&plan, 0..100)?;

        assert_eq!(ctx.leaf_work_count(), 2);
        let info_morsel = ctx.leaf_work()[1].morsel();
        let steps = ctx.drain_steps();
        assert!(matches!(
            steps.first(),
            Some(SchedulerStep::Advanced {
                morsel_id,
                ..
            }) if *morsel_id == info_morsel
        ));
        Ok(())
    }

    #[test]
    fn chunked_layout_lowering_preserves_global_order_ranges() -> VortexResult<()> {
        let first: LayoutPlanRef = Arc::new(TestLeaf::new("first", i64_dtype(), 10));
        let second: LayoutPlanRef = Arc::new(TestLeaf::new("second", i64_dtype(), 20));
        let chunked = ChunkedPlan::new(
            vec![Arc::clone(&first), Arc::clone(&second)],
            vec![0, 10, 30],
            i64_dtype(),
        );
        let ctx = lower_to_single_scheduler(&chunked, 5..25)?;

        assert_eq!(ctx.leaf_work_count(), 2);
        assert_eq!(ctx.leaf_work()[0].local_range(), &(5..10));
        assert_eq!(ctx.leaf_work()[0].global_range(), &(5..10));
        assert_eq!(ctx.leaf_work()[1].local_range(), &(0..15));
        assert_eq!(ctx.leaf_work()[1].global_range(), &(10..25));
        Ok(())
    }
}
