// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 vtable support for dictionary layouts.
//!
//! Value reads use the dictionary value domain: values read once per query and
//! cached, codes read per range (selection-aware), the pair rebuilt as a lazy
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

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::match_each_integer_ptype;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_scan::plan::DeferredReadTask;
use vortex_scan::plan::OwnedRowScope;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedRead;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStateKey;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ReadStep;
use vortex_scan::plan::ReadTask;
use vortex_scan::plan::ReadTaskOutput;
use vortex_scan::plan::RowScope;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanState;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::default_try_push_expr;
use vortex_scan::plan::request::ScanRequest;
use vortex_scan::read::ScanIoPhase;

use crate::layout_v2::Layout;
use crate::layout_v2::LayoutScanPlanCtx;
use crate::layouts_v2::dict::Dict;

const DENSE_REMAP_MAX_VALUES: usize = 1 << 20;
const DENSE_REMAP_VALUES_PER_CODE: usize = 4;
const UNREFERENCED_VALUE: usize = usize::MAX;

pub(crate) fn new_scan_plan(
    layout: Layout<Dict>,
    _req: &mut ScanRequest,
    ctx: &LayoutScanPlanCtx,
) -> VortexResult<ScanPlanRef> {
    let values = layout.child(0)?;
    let codes = layout.child(1)?;
    Ok(Arc::new(DictScanPlan {
        values_len: values.row_count(),
        dtype: layout.dtype().clone(),
        row_count: layout.row_count(),
        // Values and codes live in other row domains.
        values: values.new_scan_plan(&mut ScanRequest::empty(), ctx)?,
        codes: codes.new_scan_plan(&mut ScanRequest::empty(), ctx)?,
    }))
}

/// Reads a dict layout: shared values (another row domain, read once per
/// query) plus a codes chain in this node's row domain.
pub struct DictScanPlan {
    values: ScanPlanRef,
    values_len: u64,
    codes: ScanPlanRef,
    dtype: DType,
    row_count: u64,
}

/// Per-query dictionary caches for value-domain expression results.
#[derive(Clone)]
pub struct DictScanState {
    shared: DictSharedState,
}

#[derive(Clone)]
struct DictSharedState {
    value_exprs: Arc<Mutex<FxHashMap<Expression, Option<ArrayRef>>>>,
}

impl DictScanState {
    fn new() -> Self {
        Self {
            shared: DictSharedState::default(),
        }
    }
}

impl Default for DictSharedState {
    fn default() -> Self {
        Self {
            value_exprs: Arc::new(Mutex::new(FxHashMap::default())),
        }
    }
}

/// A pushed scalar expression over a dictionary value.
struct DictExprScanPlan {
    dict: Arc<DictScanPlan>,
    expr: Expression,
    dtype: DType,
}

struct DictPreparedRead {
    node: Arc<DictScanPlan>,
    values_read: PreparedReadRef,
    codes_read: PreparedReadRef,
}

struct DictExprPreparedRead {
    node: Arc<DictExprScanPlan>,
    state: Arc<DictScanState>,
    values_read: PreparedReadRef,
    codes_read: PreparedReadRef,
}

fn value_expr_is_expensive(expr: &Expression) -> bool {
    // TODO: Move this cost classification onto ScalarFnVTable instead of matching function IDs
    // here.
    matches!(
        expr.id().as_str(),
        "vortex.like"
            | "vortex.list.contains"
            | "vortex.dynamic"
            | "vortex.variant_get"
            | "vortex.parquet.variant"
    ) || expr.children().iter().any(value_expr_is_expensive)
}

fn sparse_dict_candidate(values_len: u64, rows: RowScope<'_>) -> bool {
    rows.demands_all_selected()
        && !rows.selection.all_true()
        && rows.selection.density() < 0.5
        && matches!(
            usize::try_from(values_len),
            Ok(values_len) if values_len > rows.demand.true_count()
        )
}

fn sparse_value_expr_candidate(expr: &Expression, values_len: u64, rows: RowScope<'_>) -> bool {
    sparse_dict_candidate(values_len, rows) && value_expr_is_expensive(expr)
}

fn value_expr_candidate(expr: &Expression, values_len: u64, rows: RowScope<'_>) -> bool {
    if sparse_value_expr_candidate(expr, values_len, rows) {
        return false;
    }
    if !value_expr_is_expensive(expr) {
        return true;
    }

    let Ok(values_len) = usize::try_from(values_len) else {
        return false;
    };
    let demand = rows.demand.true_count();
    // Dense scans will usually touch every morsel in this dictionary. Since value-domain
    // expressions are cached per DictScanState, allow a small amount of look-ahead instead of
    // repeatedly evaluating expensive predicates over decoded row values.
    values_len <= demand
        || (rows.selection.all_true()
            && rows.demand.all_true()
            && values_len <= demand.saturating_mul(4))
}

impl DictScanPlan {
    fn build_dict(&self, codes: ArrayRef, values: ArrayRef) -> VortexResult<ArrayRef> {
        // SAFETY: the codes and values children come from a validated dictionary layout.
        Ok(unsafe { DictArray::new_unchecked(codes, values) }.into_array())
    }
}

impl ScanPlan for DictScanPlan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(DictScanState::new()))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        if is_root(expr) {
            Ok(Some(self))
        } else {
            let dtype = expr.return_dtype(&self.dtype)?;
            Ok(Some(Arc::new(DictExprScanPlan {
                dict: self,
                expr: expr.clone(),
                dtype,
            })))
        }
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.codes.split_hints()
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let values_read = Arc::clone(&self.values)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary values did not produce a prepared read"))?;
        let codes_read = Arc::clone(&self.codes)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary codes did not produce a prepared read"))?;
        Ok(Some(Arc::new(DictPreparedRead {
            node: self,
            values_read,
            codes_read,
        })))
    }

    /// Codes live in this node's row domain and release with it.
    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let _ = (frontier, state);
        Ok(())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dict(")?;
        self.codes.fmt_chain(f)?;
        write!(f, ")")
    }
}

impl ScanPlan for DictExprScanPlan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.dict.row_count
    }

    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(DictScanState::new()))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        default_try_push_expr(self, expr)
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let key =
            PreparedStateKey::new::<DictScanState>(Arc::as_ptr(&self.dict) as *const () as usize);
        let state = cx.shared_state(key, || Ok(DictScanState::new()))?;
        let values_read = Arc::clone(&self.dict.values)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary values did not produce a prepared read"))?;
        let codes_read = Arc::clone(&self.dict.codes)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary codes did not produce a prepared read"))?;
        Ok(Some(Arc::new(DictExprPreparedRead {
            node: self,
            state,
            values_read,
            codes_read,
        })))
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        self.dict.release(frontier, state)
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dict_expr({})", self.expr)
    }
}

enum DictReadState {
    Start,
    SparseValues {
        compact_codes: ArrayRef,
        values: Option<Box<dyn ReadTask>>,
    },
    FullValues {
        codes: ArrayRef,
        values: Option<Box<dyn ReadTask>>,
    },
}

struct DictReadTask {
    read: Arc<DictPreparedRead>,
    codes: Box<dyn ReadTask>,
    phase: ScanIoPhase,
    state: DictReadState,
}

impl ReadTask for DictReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let task = *self;
        match task.state {
            DictReadState::Start => {
                let DictReadTask {
                    read,
                    codes,
                    phase,
                    state: _,
                } = task;
                let codes_step = codes.into_step()?;
                let values_prefetch_step =
                    DictReadTask::create_full_values_task_for(&read, phase)?.into_step()?;
                let mut prefetch_reads = codes_step.prefetch_reads;
                prefetch_reads.extend(values_prefetch_step.required_reads);
                prefetch_reads.extend(values_prefetch_step.prefetch_reads);
                Ok(ReadStep::new(
                    codes_step.required_reads,
                    prefetch_reads,
                    move |io, local, results| match codes_step
                        .continuation
                        .run(io, local, results)?
                    {
                        ReadTaskOutput::Continue(codes) => {
                            Ok(ReadTaskOutput::Continue(Box::new(DictReadTask {
                                read,
                                codes,
                                phase,
                                state: DictReadState::Start,
                            })))
                        }
                        ReadTaskOutput::Ready(codes) => {
                            let mut task = DictReadTask {
                                read,
                                codes: Box::new(DeferredReadTask),
                                phase,
                                state: DictReadState::Start,
                            };
                            let rows = OwnedRowScope::selected(Mask::new_true(codes.len()));
                            if sparse_dict_candidate(task.read.node.values_len, rows.as_scope()) {
                                let values_len = usize::try_from(task.read.node.values_len)
                                    .map_err(|_| {
                                        vortex_err!("dictionary values length exceeds usize")
                                    })?;
                                if let Some((compact_codes, value_selection)) =
                                    compact_codes_and_value_selection(
                                        codes.clone(),
                                        values_len,
                                        local,
                                    )?
                                {
                                    let values = task
                                        .create_values_task(RowScope::selected(&value_selection))?;
                                    task.state = DictReadState::SparseValues {
                                        compact_codes,
                                        values: Some(values),
                                    };
                                    return Ok(ReadTaskOutput::Continue(Box::new(task)));
                                }
                            }
                            let values = task.create_full_values_task()?;
                            task.state = DictReadState::FullValues {
                                codes,
                                values: Some(values),
                            };
                            Ok(ReadTaskOutput::Continue(Box::new(task)))
                        }
                    },
                ))
            }
            DictReadState::SparseValues {
                compact_codes,
                mut values,
            } => {
                let values_task = values.take().ok_or_else(|| {
                    vortex_err!("dictionary sparse values task was not initialized")
                })?;
                let values_step = values_task.into_step()?;
                let read = task.read;
                let phase = task.phase;
                Ok(ReadStep::new(
                    values_step.required_reads,
                    values_step.prefetch_reads,
                    move |io, local, results| match values_step
                        .continuation
                        .run(io, local, results)?
                    {
                        ReadTaskOutput::Continue(values) => {
                            Ok(ReadTaskOutput::Continue(Box::new(DictReadTask {
                                read,
                                codes: Box::new(DeferredReadTask),
                                phase,
                                state: DictReadState::SparseValues {
                                    compact_codes,
                                    values: Some(values),
                                },
                            })))
                        }
                        ReadTaskOutput::Ready(values) => Ok(ReadTaskOutput::Ready(
                            read.node.build_dict(compact_codes, values)?.optimize()?,
                        )),
                    },
                ))
            }
            DictReadState::FullValues { codes, mut values } => {
                let values_task = values.take().ok_or_else(|| {
                    vortex_err!("dictionary full values task was not initialized")
                })?;
                let values_step = values_task.into_step()?;
                let read = task.read;
                let phase = task.phase;
                Ok(ReadStep::new(
                    values_step.required_reads,
                    values_step.prefetch_reads,
                    move |io, local, results| match values_step
                        .continuation
                        .run(io, local, results)?
                    {
                        ReadTaskOutput::Continue(values) => {
                            Ok(ReadTaskOutput::Continue(Box::new(DictReadTask {
                                read,
                                codes: Box::new(DeferredReadTask),
                                phase,
                                state: DictReadState::FullValues {
                                    codes,
                                    values: Some(values),
                                },
                            })))
                        }
                        ReadTaskOutput::Ready(values) => Ok(ReadTaskOutput::Ready(
                            read.node.build_dict(codes, values)?.optimize()?,
                        )),
                    },
                ))
            }
        }
    }
}
impl DictReadTask {
    fn create_values_task(&mut self, rows: RowScope<'_>) -> VortexResult<Box<dyn ReadTask>> {
        Self::create_values_task_for(&self.read, self.phase, rows)
    }

    fn create_values_task_for(
        read: &Arc<DictPreparedRead>,
        phase: ScanIoPhase,
        rows: RowScope<'_>,
    ) -> VortexResult<Box<dyn ReadTask>> {
        let range = 0..read.node.values_len;
        let owned_rows = OwnedRowScope::try_new(rows.selection.clone(), rows.demand.clone())?;
        Arc::clone(&read.values_read).create_task(range, owned_rows, phase)
    }

    fn create_full_values_task(&mut self) -> VortexResult<Box<dyn ReadTask>> {
        Self::create_full_values_task_for(&self.read, self.phase)
    }

    fn create_full_values_task_for(
        read: &Arc<DictPreparedRead>,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        let values_selection = Mask::new_true(
            usize::try_from(read.node.values_len)
                .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?,
        );
        Self::create_values_task_for(read, phase, RowScope::selected(&values_selection))
    }
}

impl PreparedRead for DictPreparedRead {
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        Ok(Box::new(DictReadTask {
            codes: Arc::clone(&self.codes_read).create_task(range, rows, phase)?,
            read: self,
            phase,
            state: DictReadState::Start,
        }))
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.codes_read.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

fn compact_codes_and_value_selection(
    codes: ArrayRef,
    values_len: usize,
    local: &mut ExecutionCtx,
) -> VortexResult<Option<(ArrayRef, Mask)>> {
    let codes = codes.execute::<PrimitiveArray>(local)?;
    let validity = codes.validity()?;
    let valid = validity.execute_mask(codes.len(), local)?;
    if valid.all_false() {
        return Ok(None);
    }

    match_each_integer_ptype!(codes.ptype(), |Code| {
        compact_codes_and_value_selection_typed::<Code>(
            codes.as_slice::<Code>(),
            validity,
            &valid,
            values_len,
        )
    })
}

fn compact_codes_and_value_selection_typed<Code>(
    codes: &[Code],
    validity: Validity,
    valid: &Mask,
    values_len: usize,
) -> VortexResult<Option<(ArrayRef, Mask)>>
where
    Code: NativePType + TryFrom<usize>,
    usize: TryFrom<Code>,
{
    if use_dense_value_rank_map(codes.len(), valid.true_count(), values_len) {
        return compact_codes_and_value_selection_dense(codes, validity, valid, values_len);
    }

    let referenced = referenced_values(codes, valid, values_len)?;
    if referenced.is_empty() || referenced.len() == values_len {
        return Ok(None);
    }

    let compact = remap_codes(codes, valid, values_len, &referenced)?;
    let value_selection = Mask::from_indices(values_len, referenced);
    let compact_codes = PrimitiveArray::new(compact.freeze(), validity).into_array();
    Ok(Some((compact_codes, value_selection)))
}

fn use_dense_value_rank_map(codes_len: usize, valid_count: usize, values_len: usize) -> bool {
    values_len <= DENSE_REMAP_MAX_VALUES
        && values_len <= valid_count.saturating_mul(DENSE_REMAP_VALUES_PER_CODE)
        && values_len <= codes_len.saturating_mul(DENSE_REMAP_VALUES_PER_CODE)
}

fn compact_codes_and_value_selection_dense<Code>(
    codes: &[Code],
    validity: Validity,
    valid: &Mask,
    values_len: usize,
) -> VortexResult<Option<(ArrayRef, Mask)>>
where
    Code: NativePType + TryFrom<usize>,
    usize: TryFrom<Code>,
{
    let mut rank_by_value = vec![UNREFERENCED_VALUE; values_len];
    mark_referenced_values(codes, valid, values_len, &mut rank_by_value)?;

    let mut referenced = Vec::with_capacity(valid.true_count().min(values_len));
    let mut rank = 0;
    for (value_idx, value_rank) in rank_by_value.iter_mut().enumerate() {
        if *value_rank != UNREFERENCED_VALUE {
            *value_rank = rank;
            referenced.push(value_idx);
            rank += 1;
        }
    }

    if referenced.is_empty() || referenced.len() == values_len {
        return Ok(None);
    }

    let compact = remap_codes_dense(codes, valid, values_len, &rank_by_value)?;
    let value_selection = Mask::from_indices(values_len, referenced);
    let compact_codes = PrimitiveArray::new(compact.freeze(), validity).into_array();
    Ok(Some((compact_codes, value_selection)))
}

fn mark_referenced_values<Code>(
    codes: &[Code],
    valid: &Mask,
    values_len: usize,
    rank_by_value: &mut [usize],
) -> VortexResult<()>
where
    Code: Copy + fmt::Display,
    usize: TryFrom<Code>,
{
    match valid.bit_buffer() {
        AllOr::All => {
            for &code in codes {
                let idx = checked_code_index(code, values_len)?;
                rank_by_value[idx] = 0;
            }
        }
        AllOr::None => {}
        AllOr::Some(mask) => {
            for idx in mask.set_indices() {
                let value_idx = checked_code_index(codes[idx], values_len)?;
                rank_by_value[value_idx] = 0;
            }
        }
    }
    Ok(())
}

fn referenced_values<Code>(
    codes: &[Code],
    valid: &Mask,
    values_len: usize,
) -> VortexResult<Vec<usize>>
where
    Code: Copy + fmt::Display,
    usize: TryFrom<Code>,
{
    let mut referenced = Vec::with_capacity(valid.true_count().min(values_len));
    match valid.bit_buffer() {
        AllOr::All => {
            for &code in codes {
                referenced.push(checked_code_index(code, values_len)?);
            }
        }
        AllOr::None => {}
        AllOr::Some(mask) => {
            for idx in mask.set_indices() {
                referenced.push(checked_code_index(codes[idx], values_len)?);
            }
        }
    }
    referenced.sort_unstable();
    referenced.dedup();
    Ok(referenced)
}

fn remap_codes<Code>(
    codes: &[Code],
    valid: &Mask,
    values_len: usize,
    referenced: &[usize],
) -> VortexResult<BufferMut<Code>>
where
    Code: Copy + Default + fmt::Display + TryFrom<usize>,
    usize: TryFrom<Code>,
{
    let mut compact = BufferMut::<Code>::with_capacity(codes.len());
    match valid.bit_buffer() {
        AllOr::All => {
            for &code in codes {
                compact.push(compact_code(code, values_len, referenced)?);
            }
        }
        AllOr::None => compact.extend(std::iter::repeat_n(Code::default(), codes.len())),
        AllOr::Some(mask) => {
            let mut valid_indices = mask.set_indices();
            let mut next_valid = valid_indices.next();
            for (idx, &code) in codes.iter().enumerate() {
                if next_valid == Some(idx) {
                    compact.push(compact_code(code, values_len, referenced)?);
                    next_valid = valid_indices.next();
                } else {
                    compact.push(Code::default());
                }
            }
        }
    }
    Ok(compact)
}

fn remap_codes_dense<Code>(
    codes: &[Code],
    valid: &Mask,
    values_len: usize,
    rank_by_value: &[usize],
) -> VortexResult<BufferMut<Code>>
where
    Code: Copy + Default + fmt::Display + TryFrom<usize>,
    usize: TryFrom<Code>,
{
    let mut compact = BufferMut::<Code>::with_capacity(codes.len());
    match valid.bit_buffer() {
        AllOr::All => {
            for &code in codes {
                compact.push(compact_code_dense(code, values_len, rank_by_value)?);
            }
        }
        AllOr::None => compact.extend(std::iter::repeat_n(Code::default(), codes.len())),
        AllOr::Some(mask) => {
            let mut valid_indices = mask.set_indices();
            let mut next_valid = valid_indices.next();
            for (idx, &code) in codes.iter().enumerate() {
                if next_valid == Some(idx) {
                    compact.push(compact_code_dense(code, values_len, rank_by_value)?);
                    next_valid = valid_indices.next();
                } else {
                    compact.push(Code::default());
                }
            }
        }
    }
    Ok(compact)
}

fn checked_code_index<Code>(code: Code, values_len: usize) -> VortexResult<usize>
where
    Code: Copy + fmt::Display,
    usize: TryFrom<Code>,
{
    let idx = usize::try_from(code)
        .map_err(|_| vortex_err!("invalid negative dictionary code {code}"))?;
    if idx >= values_len {
        vortex_bail!(
            "dictionary code {idx} out of bounds for values length {}",
            values_len
        );
    }
    Ok(idx)
}

fn compact_code_dense<Code>(
    code: Code,
    values_len: usize,
    rank_by_value: &[usize],
) -> VortexResult<Code>
where
    Code: Copy + fmt::Display + TryFrom<usize>,
    usize: TryFrom<Code>,
{
    let idx = checked_code_index(code, values_len)?;
    let rank = rank_by_value[idx];
    if rank == UNREFERENCED_VALUE {
        vortex_bail!("dictionary code {idx} missing from sparse referenced value map");
    }
    Code::try_from(rank).map_err(|_| {
        vortex_err!(
            "sparse dictionary code rank {rank} cannot be represented by original code type"
        )
    })
}

fn compact_code<Code>(code: Code, values_len: usize, referenced: &[usize]) -> VortexResult<Code>
where
    Code: Copy + fmt::Display + TryFrom<usize>,
    usize: TryFrom<Code>,
{
    let idx = checked_code_index(code, values_len)?;
    let rank = referenced.binary_search(&idx).map_err(|_| {
        vortex_err!("dictionary code {idx} missing from sparse referenced value set")
    })?;
    Code::try_from(rank).map_err(|_| {
        vortex_err!(
            "sparse dictionary code rank {rank} cannot be represented by original code type"
        )
    })
}

enum DictExprReadState {
    Start,
    Values {
        codes: ArrayRef,
        values: Option<Box<dyn ReadTask>>,
        mode: DictExprValueMode,
    },
}

enum DictExprValueMode {
    Full { try_value_expr: bool },
    Sparse { compact_codes: ArrayRef },
}

struct DictExprReadTask {
    read: Arc<DictExprPreparedRead>,
    codes: Box<dyn ReadTask>,
    phase: ScanIoPhase,
    state: DictExprReadState,
}

impl ReadTask for DictExprReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let task = *self;
        match task.state {
            DictExprReadState::Start => {
                let DictExprReadTask {
                    read,
                    codes,
                    phase,
                    state: _,
                } = task;
                let codes_step = codes.into_step()?;
                let values_prefetch_step =
                    DictExprReadTask::create_full_values_task_for(&read, phase)?.into_step()?;
                let mut prefetch_reads = codes_step.prefetch_reads;
                prefetch_reads.extend(values_prefetch_step.required_reads);
                prefetch_reads.extend(values_prefetch_step.prefetch_reads);
                Ok(ReadStep::new(
                    codes_step.required_reads,
                    prefetch_reads,
                    move |io, local, results| match codes_step
                        .continuation
                        .run(io, local, results)?
                    {
                        ReadTaskOutput::Continue(codes) => {
                            Ok(ReadTaskOutput::Continue(Box::new(DictExprReadTask {
                                read,
                                codes,
                                phase,
                                state: DictExprReadState::Start,
                            })))
                        }
                        ReadTaskOutput::Ready(codes) => {
                            let mut task = DictExprReadTask {
                                read,
                                codes: Box::new(DeferredReadTask),
                                phase,
                                state: DictExprReadState::Start,
                            };
                            let selection = Mask::new_true(codes.len());
                            let rows = RowScope::selected(&selection);
                            let sparse_candidate = sparse_value_expr_candidate(
                                &task.read.node.expr,
                                task.read.node.dict.values_len,
                                rows,
                            );
                            let value_candidate = value_expr_candidate(
                                &task.read.node.expr,
                                task.read.node.dict.values_len,
                                rows,
                            );
                            let all_valid = !codes.dtype().is_nullable()
                                || codes
                                    .validity()
                                    .and_then(|validity| validity.execute_mask(codes.len(), local))?
                                    .all_true();
                            let mut try_value_expr = value_candidate && all_valid;
                            if try_value_expr {
                                let cached = task
                                    .read
                                    .state
                                    .shared
                                    .value_exprs
                                    .lock()
                                    .get(&task.read.node.expr)
                                    .cloned();
                                match cached {
                                    Some(Some(value_expr)) => {
                                        return Ok(ReadTaskOutput::Ready(
                                            task.read.node.dict.build_dict(codes, value_expr)?,
                                        ));
                                    }
                                    Some(None) => try_value_expr = false,
                                    None => {}
                                }
                            }
                            if try_value_expr {
                                let values = task.create_full_values_task()?;
                                task.state = DictExprReadState::Values {
                                    codes,
                                    values: Some(values),
                                    mode: DictExprValueMode::Full {
                                        try_value_expr: true,
                                    },
                                };
                                return Ok(ReadTaskOutput::Continue(Box::new(task)));
                            }
                            if sparse_candidate {
                                let values_len = usize::try_from(task.read.node.dict.values_len)
                                    .map_err(|_| {
                                        vortex_err!("dictionary values length exceeds usize")
                                    })?;
                                if let Some((compact_codes, value_selection)) =
                                    compact_codes_and_value_selection(
                                        codes.clone(),
                                        values_len,
                                        local,
                                    )?
                                {
                                    let values = task
                                        .create_values_task(RowScope::selected(&value_selection))?;
                                    task.state = DictExprReadState::Values {
                                        codes,
                                        values: Some(values),
                                        mode: DictExprValueMode::Sparse { compact_codes },
                                    };
                                    return Ok(ReadTaskOutput::Continue(Box::new(task)));
                                }
                            }
                            let values = task.create_full_values_task()?;
                            task.state = DictExprReadState::Values {
                                codes,
                                values: Some(values),
                                mode: DictExprValueMode::Full {
                                    try_value_expr: false,
                                },
                            };
                            Ok(ReadTaskOutput::Continue(Box::new(task)))
                        }
                    },
                ))
            }
            DictExprReadState::Values {
                codes,
                mut values,
                mode,
            } => {
                let values_task = values.take().ok_or_else(|| {
                    vortex_err!("dictionary expression values task was not initialized")
                })?;
                let values_step = values_task.into_step()?;
                let read = task.read;
                let phase = task.phase;
                Ok(ReadStep::new(
                    values_step.required_reads,
                    values_step.prefetch_reads,
                    move |io, local, results| match values_step
                        .continuation
                        .run(io, local, results)?
                    {
                        ReadTaskOutput::Continue(values) => {
                            Ok(ReadTaskOutput::Continue(Box::new(DictExprReadTask {
                                read,
                                codes: Box::new(DeferredReadTask),
                                phase,
                                state: DictExprReadState::Values {
                                    codes,
                                    values: Some(values),
                                    mode,
                                },
                            })))
                        }
                        ReadTaskOutput::Ready(values_array) => {
                            finish_dict_expr_values(read, phase, codes, mode, values_array, local)
                        }
                    },
                ))
            }
        }
    }
}

fn finish_dict_expr_values(
    read: Arc<DictExprPreparedRead>,
    phase: ScanIoPhase,
    codes: ArrayRef,
    mode: DictExprValueMode,
    values_array: ArrayRef,
    local: &mut ExecutionCtx,
) -> VortexResult<ReadTaskOutput> {
    match mode {
        DictExprValueMode::Full { try_value_expr } => {
            if try_value_expr {
                let value_expr = {
                    let mut value_exprs = read.state.shared.value_exprs.lock();
                    if let Some(cached) = value_exprs.get(&read.node.expr).cloned() {
                        cached
                    } else {
                        let computed = values_array.clone().apply(&read.node.expr).and_then(
                            |array| match array.clone().execute::<Mask>(local) {
                                Ok(mask) => {
                                    let DType::Bool(nullability) = array.dtype() else {
                                        return array.execute::<ArrayRef>(local);
                                    };
                                    Ok(BoolArray::new(
                                        mask.to_bit_buffer(),
                                        Validity::from(nullability),
                                    )
                                    .into_array())
                                }
                                Err(_) => array.execute::<ArrayRef>(local),
                            },
                        );
                        let value_expr = match computed {
                            Ok(array) => Some(array),
                            Err(error) => {
                                tracing::debug!(
                                    predicate = %read.node.expr,
                                    %error,
                                    "dict value-domain expression read unavailable"
                                );
                                None
                            }
                        };
                        value_exprs.insert(read.node.expr.clone(), value_expr.clone());
                        value_expr
                    }
                };
                if let Some(value_expr) = value_expr {
                    return Ok(ReadTaskOutput::Ready(
                        read.node.dict.build_dict(codes, value_expr)?,
                    ));
                }
            }
            let input = read.node.dict.build_dict(codes, values_array)?.optimize()?;
            Ok(ReadTaskOutput::Ready(
                input.apply(&read.node.expr)?.execute::<ArrayRef>(local)?,
            ))
        }
        DictExprValueMode::Sparse { compact_codes } => {
            let input = read
                .node
                .dict
                .build_dict(compact_codes, values_array)?
                .optimize()?;
            let computed = input
                .apply(&read.node.expr)
                .and_then(|array| array.execute::<ArrayRef>(local));
            match computed {
                Ok(array) => Ok(ReadTaskOutput::Ready(array)),
                Err(error) => {
                    tracing::debug!(
                        predicate = %read.node.expr,
                        %error,
                        "sparse dict expression read unavailable"
                    );
                    let full_values = DictExprReadTask::create_full_values_task_for(&read, phase)?;
                    Ok(ReadTaskOutput::Continue(Box::new(DictExprReadTask {
                        read,
                        codes: Box::new(DeferredReadTask),
                        phase,
                        state: DictExprReadState::Values {
                            codes,
                            values: Some(full_values),
                            mode: DictExprValueMode::Full {
                                try_value_expr: false,
                            },
                        },
                    })))
                }
            }
        }
    }
}

impl DictExprReadTask {
    fn create_values_task(&mut self, rows: RowScope<'_>) -> VortexResult<Box<dyn ReadTask>> {
        Self::create_values_task_for(&self.read, self.phase, rows)
    }

    fn create_values_task_for(
        read: &Arc<DictExprPreparedRead>,
        phase: ScanIoPhase,
        rows: RowScope<'_>,
    ) -> VortexResult<Box<dyn ReadTask>> {
        let range = 0..read.node.dict.values_len;
        let owned_rows = OwnedRowScope::try_new(rows.selection.clone(), rows.demand.clone())?;
        Arc::clone(&read.values_read).create_task(range, owned_rows, phase)
    }

    fn create_full_values_task(&mut self) -> VortexResult<Box<dyn ReadTask>> {
        Self::create_full_values_task_for(&self.read, self.phase)
    }

    fn create_full_values_task_for(
        read: &Arc<DictExprPreparedRead>,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        let values_selection = Mask::new_true(
            usize::try_from(read.node.dict.values_len)
                .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?,
        );
        Self::create_values_task_for(read, phase, RowScope::selected(&values_selection))
    }
}

impl PreparedRead for DictExprPreparedRead {
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        Ok(Box::new(DictExprReadTask {
            codes: Arc::clone(&self.codes_read).create_task(range, rows, phase)?,
            read: self,
            phase,
            state: DictExprReadState::Start,
        }))
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.codes_read.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::compact_codes_and_value_selection_typed;

    #[test]
    fn dense_compaction_preserves_sparse_value_order_and_validity() -> VortexResult<()> {
        let validity = Validity::from_iter([true, false, true, true, true, true]);
        let valid = validity.execute_mask(6, &mut LEGACY_SESSION.create_execution_ctx())?;
        let (compact_codes, value_selection) = compact_codes_and_value_selection_typed::<u8>(
            &[7, 9, 3, 7, 1, 3],
            validity,
            &valid,
            8,
        )?
        .expect("sparse dict compaction should be available");

        assert_eq!(value_selection, Mask::from_indices(8, [1, 3, 7]));
        let compact_codes =
            compact_codes.execute::<PrimitiveArray>(&mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(compact_codes.as_slice::<u8>(), &[2, 0, 1, 2, 0, 1]);
        assert_eq!(
            compact_codes
                .validity()?
                .execute_mask(6, &mut LEGACY_SESSION.create_execution_ctx())?,
            Mask::from_indices(6, [0, 2, 3, 4, 5])
        );

        Ok(())
    }

    #[test]
    fn dense_compaction_returns_none_when_all_values_referenced() -> VortexResult<()> {
        let validity = Validity::NonNullable;
        let valid = validity.execute_mask(4, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert!(
            compact_codes_and_value_selection_typed::<u8>(
                buffer![2u8, 0, 1, 3].as_slice(),
                validity,
                &valid,
                4,
            )?
            .is_none()
        );

        Ok(())
    }
}
