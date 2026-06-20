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

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SharedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::match_each_integer_ptype;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_scan::plan::FileReader;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedRead;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStateKey;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::RowScope;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanState;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::default_try_push_expr;
use vortex_scan::plan::request::ScanRequest;
use vortex_session::VortexSession;

use crate::layout_v2::Dict;
use crate::layout_v2::Layout;
use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

pub(crate) fn new_scan_plan(
    layout: Layout<Dict>,
    _req: &mut ScanRequest,
    session: &VortexSession,
) -> VortexResult<ScanPlanRef> {
    let values = layout.child(0)?;
    let codes = layout.child(1)?;
    Ok(Arc::new(DictScanPlan {
        values_len: values.row_count(),
        // Values and codes live in other row domains.
        values: values.new_scan_plan(&mut ScanRequest::empty(), session)?,
        codes: codes.new_scan_plan(&mut ScanRequest::empty(), session)?,
    }))
}

/// Reads a dict layout: shared values (another row domain, read once per
/// query) plus a codes chain in this node's row domain.
pub struct DictScanPlan {
    values: ScanPlanRef,
    values_len: u64,
    codes: ScanPlanRef,
}

/// Per-query dictionary caches: the shared values relation and cached
/// value-domain expression results.
#[derive(Clone)]
pub struct DictScanState {
    shared: DictSharedState,
}

#[derive(Clone)]
struct DictSharedState {
    values: Arc<Mutex<Option<SharedArrayFuture>>>,
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
            values: Arc::new(Mutex::new(None)),
            value_exprs: Arc::new(Mutex::new(FxHashMap::default())),
        }
    }
}

/// A pushed scalar expression over a dictionary value.
struct DictExprScanPlan {
    dict: Arc<DictScanPlan>,
    expr: Expression,
}

struct DictPreparedRead {
    node: Arc<DictScanPlan>,
    state: Arc<DictScanState>,
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

impl DictScanPlan {
    /// The values relation wrapped in a `SharedArray`, read once per query.
    fn values(
        &self,
        values_read: PreparedReadRef,
        io: &FileReader,
        state: &DictScanState,
    ) -> SharedArrayFuture {
        if let Some(hit) = state.shared.values.lock().clone() {
            return hit;
        }

        let mut guard = state.shared.values.lock();
        if let Some(hit) = guard.clone() {
            return hit;
        }

        let values_len = self.values_len;
        let io = io.clone();
        let future = async move {
            let selection =
                Mask::new_true(usize::try_from(values_len).map_err(|_| {
                    Arc::new(vortex_err!("dictionary values length exceeds usize"))
                })?);
            let mut local = io.session().create_execution_ctx();
            let values = values_read
                .read_scoped(
                    0..values_len,
                    RowScope::selected(&selection),
                    &io,
                    &mut local,
                )
                .await
                .map_err(Arc::new)?;
            // The shared future single-flights IO. `SharedArray` separately memoizes execution of
            // the full dictionary values across batches; sparse selected reads bypass this path.
            Ok(SharedArray::new(values).into_array())
        }
        .boxed()
        .shared();

        *guard = Some(future.clone());
        future
    }

    fn build_dict(&self, codes: ArrayRef, values: ArrayRef) -> VortexResult<ArrayRef> {
        // SAFETY: the codes and values children come from a validated dictionary layout.
        Ok(unsafe { DictArray::new_unchecked(codes, values) }.into_array())
    }
}

impl ScanPlan for DictScanPlan {
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
            Ok(Some(Arc::new(DictExprScanPlan {
                dict: self,
                expr: expr.clone(),
            })))
        }
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.codes.split_hints()
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let key = PreparedStateKey::new::<DictScanState>(Arc::as_ptr(&self) as *const () as usize);
        let state = cx.shared_state(key, || Ok(DictScanState::new()))?;
        let values_read = Arc::clone(&self.values)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary values did not produce a prepared read"))?;
        let codes_read = Arc::clone(&self.codes)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("dictionary codes did not produce a prepared read"))?;
        Ok(Some(Arc::new(DictPreparedRead {
            node: self,
            state,
            values_read,
            codes_read,
        })))
    }

    /// Codes live in this node's row domain and release with it. The
    /// cached values relation and value-domain expression results stay:
    /// they are read once per query by design and consulted by every
    /// remaining morsel.
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

impl PreparedRead for DictPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            if sparse_dict_candidate(self.node.values_len, rows) {
                let codes = self
                    .codes_read
                    .read_scoped(range.clone(), rows, io, local)
                    .await?;
                let values_len = usize::try_from(self.node.values_len)
                    .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?;
                if let Some((compact_codes, value_selection)) =
                    compact_codes_and_value_selection(codes.clone(), values_len, local)?
                {
                    let values = self
                        .values_read
                        .read_scoped(
                            0..self.node.values_len,
                            RowScope::selected(&value_selection),
                            io,
                            local,
                        )
                        .await?;
                    return self.node.build_dict(compact_codes, values)?.optimize();
                }

                let values = self
                    .node
                    .values(Arc::clone(&self.values_read), io, &self.state)
                    .await
                    .map_err(VortexError::from)?;
                return self.node.build_dict(codes, values)?.optimize();
            }

            let values = async {
                self.node
                    .values(Arc::clone(&self.values_read), io, &self.state)
                    .await
                    .map_err(VortexError::from)
            };
            let codes = self.codes_read.read_scoped(range, rows, io, local);
            let (values, codes) = try_join!(values, codes)?;
            self.node.build_dict(codes, values)?.optimize()
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if sparse_dict_candidate(self.node.values_len, rows) {
            return self.codes_read.segment_requests(range, rows, cx);
        }

        let values_selection = Mask::new_true(
            usize::try_from(self.node.values_len)
                .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?,
        );
        let mut requests = self.values_read.segment_requests(
            0..self.node.values_len,
            RowScope::selected(&values_selection),
            cx,
        )?;
        if requests.is_unknown() {
            return Ok(requests);
        }
        requests.extend(self.codes_read.segment_requests(range, rows, cx)?);
        Ok(requests)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.codes_read.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl DictExprPreparedRead {
    async fn value_expr(
        &self,
        io: &FileReader,
        state: &DictScanState,
        local: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(hit) = state
            .shared
            .value_exprs
            .lock()
            .get(&self.node.expr)
            .cloned()
        {
            return Ok(hit);
        }
        let values = self
            .node
            .dict
            .values(Arc::clone(&self.values_read), io, state)
            .await
            .map_err(VortexError::from)?;
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
            .shared
            .value_exprs
            .lock()
            .insert(self.node.expr.clone(), value_expr.clone());
        Ok(value_expr)
    }

    async fn sparse_expr(
        &self,
        codes: ArrayRef,
        io: &FileReader,
        local: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let values_len = usize::try_from(self.node.dict.values_len)
            .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?;
        let Some((compact_codes, value_selection)) =
            compact_codes_and_value_selection(codes, values_len, local)?
        else {
            return Ok(None);
        };

        let values = self
            .values_read
            .read_scoped(
                0..self.node.dict.values_len,
                RowScope::selected(&value_selection),
                io,
                local,
            )
            .await?;
        let input = self
            .node
            .dict
            .build_dict(compact_codes, values)?
            .optimize()?;
        let computed = input
            .apply(&self.node.expr)
            .and_then(|array| array.execute::<ArrayRef>(local));
        match computed {
            Ok(array) => Ok(Some(array)),
            Err(error) => {
                tracing::debug!(
                    predicate = %self.node.expr,
                    %error,
                    "sparse dict expression read unavailable"
                );
                Ok(None)
            }
        }
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
    let referenced = referenced_values(codes, valid, values_len)?;
    if referenced.is_empty() || referenced.len() == values_len {
        return Ok(None);
    }

    let compact = remap_codes(codes, valid, values_len, &referenced)?;
    let value_selection = Mask::from_indices(values_len, referenced);
    let compact_codes = PrimitiveArray::new(compact.freeze(), validity).into_array();
    Ok(Some((compact_codes, value_selection)))
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

impl PreparedRead for DictExprPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let sparse_candidate =
                sparse_value_expr_candidate(&self.node.expr, self.node.dict.values_len, rows);
            let value_expr = if !sparse_candidate
                && (!value_expr_is_expensive(&self.node.expr)
                    || matches!(
                        usize::try_from(self.node.dict.values_len),
                        Ok(values_len) if values_len <= rows.demand.true_count()
                    )) {
                self.value_expr(io, &self.state, local).await?
            } else {
                None
            };
            let codes = self
                .codes_read
                .read_scoped(range.clone(), rows, io, local)
                .await?;
            if let Some(value_expr) = value_expr {
                let all_valid = !codes.dtype().is_nullable()
                    || codes
                        .validity()?
                        .execute_mask(codes.len(), local)?
                        .all_true();
                if all_valid {
                    return self.node.dict.build_dict(codes, value_expr);
                }
            }
            if sparse_candidate
                && let Some(result) = self.sparse_expr(codes.clone(), io, local).await?
            {
                return Ok(result);
            }
            let values = self
                .node
                .dict
                .values(Arc::clone(&self.values_read), io, &self.state)
                .await
                .map_err(VortexError::from)?;
            let input = self.node.dict.build_dict(codes, values)?.optimize()?;
            input.apply(&self.node.expr)?.execute::<ArrayRef>(local)
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if sparse_value_expr_candidate(&self.node.expr, self.node.dict.values_len, rows) {
            return self.codes_read.segment_requests(range, rows, cx);
        }

        let values_selection = Mask::new_true(
            usize::try_from(self.node.dict.values_len)
                .map_err(|_| vortex_err!("dictionary values length exceeds usize"))?,
        );
        let mut requests = self.values_read.segment_requests(
            0..self.node.dict.values_len,
            RowScope::selected(&values_selection),
            cx,
        )?;
        if requests.is_unknown() {
            return Ok(requests);
        }
        requests.extend(self.codes_read.segment_requests(range, rows, cx)?);
        Ok(requests)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.codes_read.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}
