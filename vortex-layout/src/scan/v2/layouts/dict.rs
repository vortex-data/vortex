// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 vtable support for dictionary layouts.
//!
//! Value reads keep the v1 shape: values read once per query and cached,
//! codes read per range (selection-aware), the pair rebuilt as a lazy
//! `DictArray`. Pushed dictionary expressions also try to evaluate the
//! expression over the dictionary values once per query, then reuse the
//! resulting value-domain array with per-range codes.
//!
//! Dictionary predicate evidence is intentionally absent for now. Without
//! zone maps or indexes, reading dictionary values speculatively can cost
//! more than it proves; exact row-domain predicate work owns the codes read.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::layout_v2::Dict;
use crate::layout_v2::Layout;
use crate::scan::v2::node::DynReadPlan;
use crate::scan::v2::node::ExpandCtx;
use crate::scan::v2::node::FileReader;
use crate::scan::v2::node::PlanCtx;
use crate::scan::v2::node::PushCtx;
use crate::scan::v2::node::ReadPlan;
use crate::scan::v2::node::ReadPlanRef;
use crate::scan::v2::node::RowScope;
use crate::scan::v2::node::ScanNode;
use crate::scan::v2::node::ScanNodeRef;
use crate::scan::v2::node::ScanStateRef;
use crate::scan::v2::node::StateCtx;
use crate::scan::v2::request::NodeRequest;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

pub(crate) fn new_scan_node(
    layout: Layout<Dict>,
    _req: &mut NodeRequest,
    cx: &ExpandCtx,
) -> VortexResult<ScanNodeRef> {
    let values = layout.child(0)?;
    let codes = layout.child(1)?;
    Ok(Arc::new(DictScanNode {
        values_len: values.row_count(),
        // Values and codes live in other row domains.
        values: cx.expand_free(&values)?,
        codes: cx.expand_free(&codes)?,
    }))
}

/// Reads a dict layout: shared values (another row domain, read once per
/// query) plus a codes chain in this node's row domain.
pub struct DictScanNode {
    values: ScanNodeRef,
    values_len: u64,
    codes: ScanNodeRef,
}

/// Per-query state: the cached values relation, the child states, and
/// cached value-domain expression results.
pub struct DictScanState {
    values: Mutex<Option<ArrayRef>>,
    values_state: ScanStateRef,
    codes_state: ScanStateRef,
    value_exprs: Mutex<FxHashMap<Expression, Option<ArrayRef>>>,
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
    values_read: ReadPlanRef,
    codes_read: ReadPlanRef,
}

fn value_expr_is_expensive(expr: &Expression) -> bool {
    matches!(
        expr.id().as_str(),
        "vortex.like"
            | "vortex.byte_length"
            | "vortex.list.contains"
            | "vortex.dynamic"
            | "vortex.variant_get"
            | "vortex.parquet.variant"
    ) || expr.children().iter().any(value_expr_is_expensive)
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

impl ScanNode for DictScanNode {
    type State = DictScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<DictScanState> {
        Ok(DictScanState {
            values: Mutex::new(None),
            values_state: cx.init_node(&self.values)?,
            codes_state: cx.init_node(&self.codes)?,
            value_exprs: Mutex::new(FxHashMap::default()),
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
    /// cached values relation and value-domain expression results stay:
    /// they are read once per query by design and consulted by every
    /// remaining morsel.
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
        let values_read = Arc::clone(&self.dict.values)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary values did not produce a read plan"))?;
        let codes_read = Arc::clone(&self.dict.codes)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary codes did not produce a read plan"))?;
        Ok(Some(Arc::new(DictExprReadPlan {
            node: self,
            values_read,
            codes_read,
        })))
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
            value_exprs: Mutex::new(FxHashMap::default()),
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

impl DictExprReadPlan {
    async fn value_expr(
        &self,
        io: &FileReader,
        state: &DictScanState,
        local: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(hit) = state.value_exprs.lock().get(&self.node.expr).cloned() {
            return Ok(hit);
        }
        let values = self
            .node
            .dict
            .values(self.values_read.as_ref(), io, state, local)
            .await?;
        let computed = values.apply(&self.node.expr).and_then(|array| {
            match array.clone().execute::<Mask>(local) {
                Ok(mask) => {
                    let DType::Bool(nullability) = array.dtype() else {
                        return array.execute::<ArrayRef>(local);
                    };
                    Ok(
                        BoolArray::new(mask.to_bit_buffer(), Validity::from(nullability))
                            .into_array(),
                    )
                }
                Err(_) => array.execute::<ArrayRef>(local),
            }
        });
        let value_expr = match computed {
            Ok(array) => Some(array),
            Err(error) => {
                tracing::debug!(
                    predicate = %self.node.expr,
                    %error,
                    "dict value-domain expression read unavailable"
                );
                None
            }
        };
        state
            .value_exprs
            .lock()
            .insert(self.node.expr.clone(), value_expr.clone());
        Ok(value_expr)
    }
}

impl ReadPlan for DictExprReadPlan {
    type State = DictScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        Ok(DictScanState {
            values: Mutex::new(None),
            values_state: self.values_read.init_state(cx)?,
            codes_state: self.codes_read.init_state(cx)?,
            value_exprs: Mutex::new(FxHashMap::default()),
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
            let value_expr = if !value_expr_is_expensive(&self.node.expr)
                || matches!(
                    usize::try_from(self.node.dict.values_len),
                    Ok(values_len) if values_len <= rows.demand.true_count()
                ) {
                self.value_expr(io, state, local).await?
            } else {
                None
            };
            let codes = self
                .codes_read
                .read_scoped(range.clone(), rows, io, state.codes_state.as_ref(), local)
                .await?;
            if let Some(value_expr) = value_expr {
                let all_valid = !codes.dtype().is_nullable()
                    || codes
                        .validity()?
                        .execute_mask(codes.len(), local)?
                        .all_true();
                if all_valid {
                    return Ok(DictArray::try_new(codes, value_expr)?.into_array());
                }
            }
            let values = self
                .node
                .dict
                .values(self.values_read.as_ref(), io, state, local)
                .await?;
            let input = DictArray::try_new(codes, values)?.into_array().optimize()?;
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
        let values_selection = Mask::new_true(
            usize::try_from(self.node.dict.values_len)
                .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?,
        );
        let mut requests = self.values_read.segment_requests(
            0..self.node.dict.values_len,
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
