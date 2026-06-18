// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 rule for dictionary layouts.
//!
//! Value reads keep the v1 shape — values read once per query and
//! cached, codes read per range (selection-aware), the pair rebuilt as a
//! lazy `DictArray`. New is the runtime value-domain rewrite (plan 017
//! SP7): pushed dictionary predicate nodes answer by evaluating the
//! predicate over the *dictionary values* once per query, then mapping
//! the per-value verdicts through the codes:
//!
//! - no value satisfies the predicate (and null does not either): the
//!   whole column is proven all-false without reading a single code;
//! - every value satisfies it: all-true the same way;
//! - otherwise the per-value mask maps through the range's codes into an
//!   exact per-row mask, costing a code read but never a value decode at
//!   data scale.
//!
//! The rewrite is exact: evaluating the predicate over the values array
//! and indexing the result by code is the same value-domain evaluation
//! vortex's expression machinery performs over a `DictArray`, including
//! null routing (a null row takes the predicate's verdict on null). A
//! predicate whose evaluation over the values fails is recorded as
//! unanswerable and falls through to residual evaluation rather than
//! failing the scan.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutEncodingId;
use crate::LayoutRef;
use crate::layouts::dict::Dict;
use crate::layouts::dict::DictLayoutEncoding;
use crate::scan::v2::evidence::EvidenceFragment;
use crate::scan::v2::evidence::PredicateEvidenceKind;
use crate::scan::v2::node::DynReadPlan;
use crate::scan::v2::node::EvidencePlan;
use crate::scan::v2::node::EvidencePlanRef;
use crate::scan::v2::node::ExpandCtx;
use crate::scan::v2::node::FileReader;
use crate::scan::v2::node::LayoutScanRule;
use crate::scan::v2::node::PlanCtx;
use crate::scan::v2::node::PushCtx;
use crate::scan::v2::node::ReadPlan;
use crate::scan::v2::node::ReadPlanRef;
use crate::scan::v2::node::RowScope;
use crate::scan::v2::node::ScanNode;
use crate::scan::v2::node::ScanNodeRef;
use crate::scan::v2::node::ScanStateCache;
use crate::scan::v2::node::ScanStateRef;
use crate::scan::v2::node::StateCtx;
use crate::scan::v2::node::read_dense;
use crate::scan::v2::request::EvidenceRequest;
use crate::scan::v2::request::NodeRequest;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

/// Scan2 rule for `vortex.dict`.
#[derive(Debug)]
pub struct DictScanRule;

impl LayoutScanRule for DictScanRule {
    type Node = DictScanNode;

    fn id(&self) -> LayoutEncodingId {
        DictLayoutEncoding.id()
    }

    fn expand(
        &self,
        layout: &LayoutRef,
        _req: &mut NodeRequest,
        cx: &ExpandCtx,
    ) -> VortexResult<DictScanNode> {
        if !layout.is::<Dict>() {
            vortex_bail!("dict scan2 rule applied to {}", layout.encoding_id());
        }
        let values = layout.child(0)?;
        let codes = layout.child(1)?;
        Ok(DictScanNode {
            dtype: layout.dtype().clone(),
            values_len: values.row_count(),
            // Values and codes live in other row domains.
            values: cx.expand_free(&values)?,
            codes: cx.expand_free(&codes)?,
        })
    }
}

/// Reads a dict layout: shared values (another row domain, read once per
/// query) plus a codes chain in this node's row domain.
pub struct DictScanNode {
    dtype: DType,
    values: ScanNodeRef,
    values_len: u64,
    codes: ScanNodeRef,
}

/// One predicate's value-domain rewrite, computed once per query.
enum ValueVerdicts {
    /// The predicate could not be evaluated over the values; produce no
    /// evidence and let residual evaluation handle it.
    Unanswerable,
    /// Per-value verdicts plus the verdict for null rows.
    Verdicts {
        /// `true` at value `v`: rows coded `v` satisfy the predicate.
        mask: Mask,
        /// Whether a null row satisfies the predicate.
        null_verdict: bool,
    },
}

/// Per-query state: the cached values relation, the child states, and
/// the per-predicate value-domain verdicts.
pub struct DictScanState {
    values: Mutex<Option<ArrayRef>>,
    values_state: ScanStateRef,
    codes_state: ScanStateRef,
    verdicts: Mutex<FxHashMap<Expression, Arc<ValueVerdicts>>>,
}

/// Planned dictionary value-domain evidence for one predicate.
struct DictEvidencePlan {
    dtype: DType,
    values_read: ReadPlanRef,
    values_len: u64,
    codes_read: ReadPlanRef,
    predicate: Expression,
}

/// A pushed scalar expression over a dictionary value.
struct DictExprScanNode {
    dict: Arc<DictScanNode>,
    expr: Expression,
}

struct DictReadPlan {
    node: Arc<DictScanNode>,
    values_read: ReadPlanRef,
    codes_read: ReadPlanRef,
}

struct DictExprReadPlan {
    node: Arc<DictExprScanNode>,
    input: ReadPlanRef,
}

impl DictScanNode {
    /// The values relation, read once per query.
    async fn values(
        &self,
        values_read: &dyn DynReadPlan,
        io: &FileReader,
        state: &DictScanState,
        local: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(hit) = state.values.lock().clone() {
            return Ok(hit);
        }
        let selection = Mask::new_true(
            usize::try_from(self.values_len)
                .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?,
        );
        let values = values_read
            .read_scoped(
                0..self.values_len,
                RowScope::selected(&selection),
                io,
                state.values_state.as_ref(),
                local,
            )
            .await?;
        *state.values.lock() = Some(values.clone());
        Ok(values)
    }
}

impl DictEvidencePlan {
    async fn values(&self, io: &FileReader, state: &DictScanState) -> VortexResult<ArrayRef> {
        if let Some(hit) = state.values.lock().clone() {
            return Ok(hit);
        }
        let values = read_dense(
            self.values_read.as_ref(),
            0..self.values_len,
            io,
            state.values_state.as_ref(),
        )
        .await?;
        *state.values.lock() = Some(values.clone());
        Ok(values)
    }

    async fn verdicts(
        &self,
        io: &FileReader,
        state: &DictScanState,
    ) -> VortexResult<Arc<ValueVerdicts>> {
        if let Some(hit) = state.verdicts.lock().get(&self.predicate) {
            return Ok(Arc::clone(hit));
        }
        let values = self.values(io, state).await?;
        let mut ctx = io.session().create_execution_ctx();
        let computed = (|| -> VortexResult<ValueVerdicts> {
            let mask = values
                .clone()
                .apply(&self.predicate)?
                .execute::<Mask>(&mut ctx)?;
            let null_verdict = if self.dtype.is_nullable() {
                let null = ConstantArray::new(Scalar::null(self.dtype.clone()), 1).into_array();
                null.apply(&self.predicate)?
                    .execute::<Mask>(&mut ctx)?
                    .value(0)
            } else {
                false
            };
            Ok(ValueVerdicts::Verdicts { mask, null_verdict })
        })();
        let verdicts = Arc::new(match computed {
            Ok(verdicts) => verdicts,
            Err(error) => {
                tracing::debug!(
                    predicate = %self.predicate,
                    %error,
                    "dict value-domain rewrite unanswerable"
                );
                ValueVerdicts::Unanswerable
            }
        });
        state
            .verdicts
            .lock()
            .insert(self.predicate.clone(), Arc::clone(&verdicts));
        Ok(verdicts)
    }
}

impl EvidencePlan for DictEvidencePlan {
    type State = DictScanState;

    fn init_state(&self, ctx: &VortexSession) -> VortexResult<DictScanState> {
        let mut cache = ScanStateCache::default();
        let mut cx = StateCtx::new(ctx, &mut cache);
        Ok(DictScanState {
            values: Mutex::new(None),
            values_state: self.values_read.init_state(&mut cx)?,
            codes_state: self.codes_read.init_state(&mut cx)?,
            verdicts: Mutex::new(FxHashMap::default()),
        })
    }

    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a FileReader,
        state: &'a DictScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>> {
        Box::pin(async move {
            let verdicts = self.verdicts(io, state).await?;
            let ValueVerdicts::Verdicts { mask, null_verdict } = verdicts.as_ref() else {
                return Ok(Vec::new());
            };
            let nullable = self.dtype.is_nullable();
            if mask.all_false() && !*null_verdict {
                return Ok(vec![EvidenceFragment::new(
                    req.range.clone(),
                    PredicateEvidenceKind::AllFalse,
                )]);
            }
            if mask.all_true() && (!nullable || *null_verdict) {
                return Ok(vec![EvidenceFragment::new(
                    req.range.clone(),
                    PredicateEvidenceKind::AllTrue,
                )]);
            }
            let codes = read_dense(
                self.codes_read.as_ref(),
                req.range.clone(),
                io,
                state.codes_state.as_ref(),
            )
            .await?;
            let mut ctx = io.session().create_execution_ctx();
            let verdict_values = BoolArray::from(mask.to_bit_buffer()).into_array();
            let mut rows = DictArray::try_new(codes.clone(), verdict_values)?
                .into_array()
                .execute::<Mask>(&mut ctx)?;
            if *null_verdict {
                let valid = codes.validity()?.execute_mask(codes.len(), &mut ctx)?;
                rows = &rows | &!valid;
            }
            Ok(vec![EvidenceFragment::new(
                req.range.clone(),
                PredicateEvidenceKind::ExactMask(rows),
            )])
        })
    }

    fn segment_requests(
        &self,
        req: &EvidenceRequest<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        let Some(verdicts) = state.verdicts.lock().get(&self.predicate).cloned() else {
            return Ok(SegmentRequests::unknown());
        };
        let ValueVerdicts::Verdicts { mask, null_verdict } = verdicts.as_ref() else {
            return Ok(SegmentRequests::none());
        };
        let nullable = self.dtype.is_nullable();
        if mask.all_false() && !*null_verdict {
            return Ok(SegmentRequests::none());
        }
        if mask.all_true() && (!nullable || *null_verdict) {
            return Ok(SegmentRequests::none());
        }
        let selection = Mask::new_true(
            usize::try_from(req.range.end - req.range.start)
                .map_err(|_| vortex_err!("dictionary evidence range exceeds usize"))?,
        );
        self.codes_read.segment_requests(
            req.range.clone(),
            RowScope::selected(&selection),
            state.codes_state.as_ref(),
            cx,
        )
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dict")
    }
}

impl ScanNode for DictScanNode {
    type State = DictScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<DictScanState> {
        Ok(DictScanState {
            values: Mutex::new(None),
            values_state: cx.init_node(&self.values)?,
            codes_state: cx.init_node(&self.codes)?,
            verdicts: Mutex::new(FxHashMap::default()),
        })
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>> {
        if is_root(expr) {
            Ok(Some(self))
        } else {
            Ok(Some(Arc::new(DictExprScanNode {
                dict: self,
                expr: expr.clone(),
            })))
        }
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.codes.split_hints()
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        let values_read = Arc::clone(&self.values)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary values did not produce a read plan"))?;
        let codes_read = Arc::clone(&self.codes)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary codes did not produce a read plan"))?;
        Ok(Some(Arc::new(DictReadPlan {
            node: self,
            values_read,
            codes_read,
        })))
    }

    /// Codes live in this node's row domain and release with it. The
    /// cached values relation and per-predicate verdicts stay — they are
    /// read once per query by design and consulted by every remaining
    /// morsel.
    fn release(&self, frontier: u64, state: &DictScanState) -> VortexResult<()> {
        self.codes.release(frontier, state.codes_state.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dict(")?;
        self.codes.fmt_chain(f)?;
        write!(f, ")")
    }
}

impl ScanNode for DictExprScanNode {
    type State = DictScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        self.dict.init_state(cx)
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        let input = Arc::clone(&self.dict).plan_read(cx)?.ok_or_else(|| {
            vortex_err!("dictionary expression input did not produce a read plan")
        })?;
        Ok(Some(Arc::new(DictExprReadPlan { node: self, input })))
    }

    fn plan_evidence(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>> {
        let values_read = Arc::clone(&self.dict.values)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary values did not produce a read plan"))?;
        let codes_read = Arc::clone(&self.dict.codes)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary codes did not produce a read plan"))?;
        Ok(vec![Arc::new(DictEvidencePlan {
            dtype: self.dict.dtype.clone(),
            values_read,
            values_len: self.dict.values_len,
            codes_read,
            predicate: self.expr.clone(),
        })])
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.dict.release(frontier, state)
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dict_expr({})", self.expr)
    }
}

impl ReadPlan for DictReadPlan {
    type State = DictScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        Ok(DictScanState {
            values: Mutex::new(None),
            values_state: self.values_read.init_state(cx)?,
            codes_state: self.codes_read.init_state(cx)?,
            verdicts: Mutex::new(FxHashMap::default()),
        })
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let values = self
                .node
                .values(self.values_read.as_ref(), io, state, local)
                .await?;
            let codes = self
                .codes_read
                .read_scoped(range, rows, io, state.codes_state.as_ref(), local)
                .await?;
            DictArray::try_new(codes, values)?.into_array().optimize()
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        let values_selection = Mask::new_true(
            usize::try_from(self.node.values_len)
                .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?,
        );
        let mut requests = self.values_read.segment_requests(
            0..self.node.values_len,
            RowScope::selected(&values_selection),
            state.values_state.as_ref(),
            cx,
        )?;
        if requests.is_unknown() {
            return Ok(requests);
        }
        requests.extend(self.codes_read.segment_requests(
            range,
            rows,
            state.codes_state.as_ref(),
            cx,
        )?);
        Ok(requests)
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.codes_read
            .release(frontier, state.codes_state.as_ref())
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl ReadPlan for DictExprReadPlan {
    type State = ScanStateRef;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        self.input.init_state(cx)
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let input = self
                .input
                .read_scoped(range, rows, io, state.as_ref(), local)
                .await?;
            input.apply(&self.node.expr)?.execute::<ArrayRef>(local)
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        self.input.segment_requests(range, rows, state.as_ref(), cx)
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.input.release(frontier, state.as_ref())
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}
