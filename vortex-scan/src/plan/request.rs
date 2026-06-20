// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runtime evidence requests for scan plans.
//!
//! Expansion produces layout-local [`ScanPlan`](super::ScanPlan)
//! trees. Predicate, projection, aggregate, and dynamic-filter handling
//! then push expressions into those plans and ask the resulting plans for
//! prepared runtime handles. Evidence requests are the per-morsel inputs to
//! those prepared evidence handles.

use std::ops::Range;

use vortex_array::expr::Expression;

use super::evidence::PredicateId;
use super::evidence::PredicateVersion;

/// Runtime evidence pass kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceMode {
    /// Normal predicate evidence collection before residual evaluation.
    Normal,
    /// Last-chance pruning immediately before projection. Only evidence
    /// plans that explicitly opt in should execute in this pass.
    RecheckBeforeProjection,
}

/// Expansion-time context reserved for layout-local scan setup.
///
/// Layout expansion does not carry predicates directly. Layout scan vtables
/// must expose expression behavior through
/// [`ScanPlan::try_push_expr`](super::ScanPlan::try_push_expr),
/// [`ScanPlan::prepare_read`](super::ScanPlan::prepare_read), and
/// [`ScanPlan::prepare_evidence`](super::ScanPlan::prepare_evidence).
#[derive(Debug, Default)]
pub struct ScanRequest;

impl ScanRequest {
    /// A request with no relation-scoped predicate payload.
    pub fn empty() -> Self {
        Self
    }
}

/// A runtime evidence request: one prepared predicate expression, scoped
/// to the producer's row domain, over one row range.
#[derive(Clone, Debug)]
pub struct OwnedEvidenceRequest {
    /// The predicate's stable id within this scan.
    pub id: PredicateId,
    /// The predicate's version.
    pub version: PredicateVersion,
    /// The predicate with `root()` rebased to the producer's rows.
    pub predicate: Expression,
    /// The rows evidence is requested for, in the producer's coordinates.
    pub range: Range<u64>,
    /// Which evidence pass is requesting fragments.
    pub mode: EvidenceMode,
}

impl OwnedEvidenceRequest {
    /// Borrow this owned request for a prepared evidence handle.
    pub fn as_request(&self) -> EvidenceRequest<'_> {
        EvidenceRequest {
            id: self.id,
            version: self.version,
            predicate: &self.predicate,
            range: self.range.clone(),
            mode: self.mode,
        }
    }
}

/// Borrowed runtime evidence request for a prepared evidence handle.
#[derive(Debug)]
pub struct EvidenceRequest<'a> {
    /// The predicate's stable id within this scan.
    pub id: PredicateId,
    /// The predicate's version (static predicates stay at
    /// [`PredicateVersion::STATIC`]; dynamic predicates move).
    pub version: PredicateVersion,
    /// The predicate with `root()` rebased to the producer's rows.
    pub predicate: &'a Expression,
    /// The rows evidence is requested for, in the producer's coordinates.
    pub range: Range<u64>,
    /// Which evidence pass is requesting fragments.
    pub mode: EvidenceMode,
}
