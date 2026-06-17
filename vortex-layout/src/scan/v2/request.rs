// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runtime evidence requests for scan2.
//!
//! Expansion produces layout-local [`ScanNode`](super::node::ScanNode)
//! trees. Predicate, projection, aggregate, and dynamic-filter handling
//! then push expressions into those nodes and ask the resulting nodes for
//! executable plans. Evidence requests are the per-morsel inputs to those
//! already-planned evidence handles.

use std::ops::Range;

use vortex_array::expr::Expression;

use crate::scan::v2::evidence::PredicateId;
use crate::scan::v2::evidence::PredicateVersion;

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
/// Scan2 no longer carries predicates through expansion. Layout rules
/// must expose expression behavior through
/// [`ScanNode::try_push_expr`](super::node::ScanNode::try_push_expr),
/// [`ScanNode::plan_read`](super::node::ScanNode::plan_read), and
/// [`ScanNode::plan_evidence`](super::node::ScanNode::plan_evidence).
#[derive(Debug, Default)]
pub struct NodeRequest;

impl NodeRequest {
    /// A request with no relation-scoped predicate payload.
    pub fn empty() -> Self {
        Self
    }
}

/// A runtime evidence request: one planned predicate expression, scoped
/// to the producer's row domain, over one row range.
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
