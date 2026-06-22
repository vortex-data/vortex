// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::transform::PartitionedExpr;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::scalar::PValue;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scan::plan::ApplyScanPlan;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedRead;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ReadContext;
use vortex_scan::plan::RowScope;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::StructValueScanPlan;
use vortex_scan::plan::default_try_push_expr;
use vortex_scan::segments::SegmentPlanCtx;
use vortex_scan::segments::SegmentRequests;
use vortex_sequence::Sequence;
use vortex_sequence::SequenceArray;

use crate::layouts::row_idx::RowIdx;
use crate::layouts::row_idx::row_idx;

pub fn with_row_idx(root: ScanPlanRef, dtype: DType, row_offset: u64) -> ScanPlanRef {
    Arc::new(RowIdxScanPlan {
        child: root,
        dtype,
        row_offset,
    })
}

struct RowIdxScanPlan {
    child: ScanPlanRef,
    dtype: DType,
    row_offset: u64,
}

enum Partitioning {
    RowIdx(Expression),
    Child(Expression),
    Partitioned(Arc<PartitionedExpr<Partition>>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Partition {
    RowIdx,
    Child,
}

impl Partition {
    fn name(&self) -> &str {
        match self {
            Partition::RowIdx => "row_idx",
            Partition::Child => "child",
        }
    }
}

impl From<Partition> for FieldName {
    fn from(value: Partition) -> Self {
        FieldName::from(value.name())
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl RowIdxScanPlan {
    fn partition_expr(&self, expr: &Expression) -> VortexResult<Partitioning> {
        if !contains_row_idx(expr) {
            return Ok(Partitioning::Child(expr.clone()));
        }

        let mut partitioned = partition(expr.clone(), &self.dtype, |expr| {
            if expr.is::<RowIdx>() {
                vec![Partition::RowIdx]
            } else if is_root(expr) {
                vec![Partition::Child]
            } else {
                vec![]
            }
        })?;

        if partitioned.partitions.len() == 1 {
            return Ok(match &partitioned.partition_annotations[0] {
                Partition::RowIdx => {
                    Partitioning::RowIdx(replace(expr.clone(), &row_idx(), root()))
                }
                Partition::Child => Partitioning::Child(expr.clone()),
            });
        }

        partitioned.partitions = partitioned
            .partitions
            .into_iter()
            .map(|p| replace(p, &row_idx(), root()))
            .collect();

        Ok(Partitioning::Partitioned(Arc::new(partitioned)))
    }
}

impl ScanPlan for RowIdxScanPlan {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        cx.init_plan(&self.child)
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        match self.partition_expr(expr)? {
            Partitioning::RowIdx(expr) => Ok(Some(Arc::new(RowIdxExprScanPlan::try_new(
                self.row_offset,
                expr,
            )?))),
            Partitioning::Child(expr) => Arc::clone(&self.child).try_push_expr(&expr, cx),
            Partitioning::Partitioned(partitioned) => {
                let mut fields = Vec::with_capacity(partitioned.partitions.len());
                for (expr, annotation) in partitioned
                    .partitions
                    .iter()
                    .zip(partitioned.partition_annotations.iter())
                {
                    let field = match annotation {
                        Partition::RowIdx => {
                            Arc::new(RowIdxExprScanPlan::try_new(self.row_offset, expr.clone())?)
                                as ScanPlanRef
                        }
                        Partition::Child => Arc::clone(&self.child)
                            .try_push_expr(expr, cx)?
                            .ok_or_else(|| {
                                vortex_error::vortex_err!(
                                    "row_idx child partition did not push expression {expr}"
                                )
                            })?,
                    };
                    fields.push(field);
                }
                let input = Arc::new(StructValueScanPlan::new(
                    partitioned.partition_names.clone(),
                    fields,
                    None,
                ));
                Ok(Some(Arc::new(ApplyScanPlan::new(
                    input,
                    partitioned.root.clone(),
                ))))
            }
        }
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        Arc::clone(&self.child).prepare_read(cx)
    }

    fn release(&self, frontier: u64, state: &vortex_scan::plan::ScanState) -> VortexResult<()> {
        self.child.release(frontier, state)
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.child.split_hints()
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row_idx:")?;
        self.child.fmt_chain(f)
    }
}

struct RowIdxExprScanPlan {
    row_offset: u64,
    expr: Expression,
    dtype: DType,
}

impl RowIdxExprScanPlan {
    fn try_new(row_offset: u64, expr: Expression) -> VortexResult<Self> {
        let dtype = expr.return_dtype(&row_idx_dtype())?;
        Ok(Self {
            row_offset,
            expr,
            dtype,
        })
    }
}

struct RowIdxPreparedRead {
    plan: Arc<RowIdxExprScanPlan>,
}

impl ScanPlan for RowIdxExprScanPlan {
    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        default_try_push_expr(self, expr)
    }

    fn prepare_read(
        self: Arc<Self>,
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedReadRef>> {
        Ok(Some(Arc::new(RowIdxPreparedRead { plan: self })))
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row_idx({})", self.expr)
    }
}

impl PreparedRead for RowIdxPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        _io: &'a ReadContext,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let dense = idx_array(self.plan.row_offset, &range).into_array();
            if rows.selection.len() != dense.len() {
                vortex_bail!(
                    "selection length {} does not match row_idx range length {}",
                    rows.selection.len(),
                    dense.len()
                );
            }
            if rows.demand.len() != dense.len() {
                vortex_bail!(
                    "demand length {} does not match row_idx range length {}",
                    rows.demand.len(),
                    dense.len()
                );
            }
            let selected = if rows.selection.all_true() {
                dense
            } else {
                dense.filter(rows.selection.clone())?
            };
            selected.apply(&self.plan.expr)?.execute::<ArrayRef>(local)
        })
    }

    fn segment_requests(
        &self,
        _range: Range<u64>,
        _rows: RowScope<'_>,
        _cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::none())
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row_idx({}) -> {}", self.plan.expr, self.plan.dtype)
    }
}

fn idx_array(row_offset: u64, row_range: &Range<u64>) -> SequenceArray {
    Sequence::try_new(
        PValue::U64(row_offset + row_range.start),
        PValue::U64(1),
        PType::U64,
        Nullability::NonNullable,
        usize::try_from(row_range.end - row_range.start)
            .vortex_expect("row range length must fit in usize"),
    )
    .vortex_expect("failed to create row index array")
}

fn row_idx_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

fn contains_row_idx(expr: &Expression) -> bool {
    expr.is::<RowIdx>() || expr.children().iter().any(contains_row_idx)
}
