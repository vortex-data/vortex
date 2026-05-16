//! Struct assembler operator.
//!
//! `StructAssembler` is the engine-visible counterpart for the
//! struct layout: the engine sees N child source operators (one per
//! field) and one `StructAssembler` that combines their batches
//! into a `StructArray`.
//!
//! For now, the assembler buffers every batch from each input,
//! waits until all inputs seal, and emits one combined struct batch
//! covering the full row range. This keeps the bookkeeping minimal
//! at the cost of memory; producing streaming aligned struct batches
//! is a follow-up that would slice each input's batches to a common
//! span on each turn.

use std::sync::Arc;
use std::task::Context;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::StructFields;
use vortex_array::validity::Validity;

use crate::Batch;
use crate::Cardinality;
use crate::Domain;
use crate::DomainId;
use crate::DomainSpan;
use crate::EngineError;
use crate::EngineResult;
use crate::InputPortId;
use crate::InputPortSpec;
use crate::Operator;
use crate::OperatorSpec;
use crate::OutputPortSpec;
use crate::RequirementCtx;
use crate::RequirementSet;
use crate::UpdateCtx;
use crate::WorkClass;
use crate::WorkConstraints;
use crate::WorkCost;
use crate::WorkCtx;
use crate::WorkKey;
use crate::WorkProposal;
use crate::WorkStatus;
use crate::WorkValue;

pub struct StructAssembler {
    label: String,
    domain: Domain,
    field_names: Vec<Arc<str>>,
    field_dtypes: Vec<DType>,
    row_count: u64,
}

pub struct StructAssemblerState {
    /// Per-input arrays observed so far.
    accumulated: Vec<Vec<ArrayRef>>,
    /// Per-input demand masks observed so far, in the same order
    /// as `accumulated`. Per-batch masks get concatenated at
    /// assembly into one mask per field, then AND'd across fields
    /// to produce the output's demand.
    accumulated_demand: Vec<Vec<vortex_mask::Mask>>,
    /// Per-input finished flag.
    finished: Vec<bool>,
    sealed: bool,
}

impl StructAssembler {
    pub fn new(
        label: impl Into<String>,
        field_names: Vec<Arc<str>>,
        field_dtypes: Vec<DType>,
        row_count: u64,
    ) -> Self {
        let label = label.into();
        let domain = Domain::new(
            DomainId::new(format!("struct_assembler:{label}")),
            Cardinality::Exact(row_count),
        );
        Self {
            label,
            domain,
            field_names,
            field_dtypes,
            row_count,
        }
    }

    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    pub fn n_fields(&self) -> usize {
        self.field_names.len()
    }
}

impl Operator for StructAssembler {
    type GlobalState = ();
    type LocalState = StructAssemblerState;

    fn spec(&self) -> OperatorSpec {
        let inputs = self
            .field_names
            .iter()
            .map(|name| InputPortSpec::new(name.as_ref(), self.domain.clone(), 1))
            .collect();
        let outputs = Some(OutputPortSpec::new(
            "out",
            self.domain.clone(),
            self.field_names.len(),
        ));
        OperatorSpec::new(self.label.clone(), inputs, outputs)
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        let n = self.field_names.len();
        Ok(StructAssemblerState {
            accumulated: (0..n).map(|_| Vec::new()).collect(),
            accumulated_demand: (0..n).map(|_| Vec::new()).collect(),
            finished: vec![false; n],
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        // Forward the output requirement unchanged to every input.
        // Use `clone_from` so each slot's existing `Vec<RowInterval>`
        // allocation is reused when capacity is sufficient.
        for slot in inputs.iter_mut() {
            slot.clone_from(output);
        }
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        // Track which inputs have sealed (read regardless of
        // proposing — the seal decision uses this on the next emit).
        for i in 0..self.field_names.len() {
            if ctx.input_finished(InputPortId::from_index(i)) {
                state.finished[i] = true;
            }
        }
        // The next emit pulls one batch from each input port and
        // assembles them into a struct. We can only emit when
        // *every* input has a batch (or has sealed). If any port is
        // empty-and-not-sealed, don't propose — the operator stays
        // idle until the missing input arrives.
        let mut min_useful_rows: Option<u64> = None;
        let mut any_data = false;
        for i in 0..self.field_names.len() {
            let port = InputPortId::from_index(i);
            match ctx.peek(port) {
                Some(batch) => {
                    any_data = true;
                    let rows = batch.demand().true_count() as u64;
                    min_useful_rows = Some(min_useful_rows.map_or(rows, |m| m.min(rows)));
                }
                None => {
                    if !ctx.input_finished(InputPortId::from_index(i)) {
                        return Ok(());
                    }
                }
            }
        }
        // All ports are either drainable or sealed.
        let value = if any_data && min_useful_rows.unwrap_or(0) > 0 {
            WorkValue::required(min_useful_rows.unwrap())
        } else {
            WorkValue::candidate(0, 0)
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            WorkClass::Emit,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::output_capacity(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        _work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        }

        // Drain whatever batches each input currently has.
        for i in 0..self.field_names.len() {
            while let Some(batch) = ctx.pop(InputPortId::from_index(i)) {
                let demand = batch.demand().clone();
                state.accumulated[i].push(batch.into_array());
                state.accumulated_demand[i].push(demand);
            };
            if ctx.input_finished(InputPortId::from_index(i)) {
                state.finished[i] = true;
            }
        }

        let all_done = state.finished.iter().all(|&f| f);
        if !all_done {
            return Ok(WorkStatus::Made);
        }

        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }

        // All inputs finished — assemble. Concatenate each field via
        // a ChunkedArray (no-op when there's only one batch).
        let mut field_arrays: Vec<ArrayRef> = Vec::with_capacity(self.field_names.len());
        for (i, dtype) in self.field_dtypes.iter().enumerate() {
            let chunks = std::mem::take(&mut state.accumulated[i]);
            let array = if chunks.is_empty() {
                return Err(EngineError::message(format!(
                    "{}: input {} produced no batches",
                    self.label, i
                )));
            } else if chunks.len() == 1 {
                let mut iter = chunks.into_iter();
                match iter.next() {
                    Some(arr) => arr,
                    None => unreachable!("len == 1 guarantees a chunk"),
                }
            } else {
                ChunkedArray::try_new(chunks, dtype.clone())
                    .map_err(|e| EngineError::message(format!("ChunkedArray::try_new: {e}")))?
                    .into_array()
            };
            field_arrays.push(array);
        }

        let names: FieldNames = FieldNames::from(
            self.field_names
                .iter()
                .map(|n| FieldName::from(n.as_ref()))
                .collect::<Vec<_>>(),
        );
        let struct_fields = StructFields::new(names, self.field_dtypes.clone());
        let row_count = usize::try_from(self.row_count).unwrap_or(usize::MAX);
        let struct_array = StructArray::try_new_with_dtype(
            field_arrays,
            struct_fields,
            row_count,
            Validity::NonNullable,
        )
        .map_err(|e| EngineError::message(format!("StructArray::try_new: {e}")))?
        .into_array();

        // Concat each field's per-batch demand into one mask per
        // field, then AND across fields. The output demand is true
        // iff every field has real data at that row.
        let mut field_masks: Vec<vortex_mask::Mask> =
            Vec::with_capacity(self.field_names.len());
        for masks in state.accumulated_demand.iter_mut() {
            let chunks = std::mem::take(masks);
            let combined = vortex_mask::Mask::concat(chunks.iter())
                .map_err(|e| EngineError::message(format!("Mask::concat: {e}")))?;
            field_masks.push(combined);
        }
        let output_demand = field_masks.into_iter().reduce(|acc, m| {
            use std::ops::BitAnd;
            acc.bitand(&m)
        }).unwrap_or_else(|| vortex_mask::Mask::new_true(
            usize::try_from(self.row_count).unwrap_or(usize::MAX),
        ));

        let batch = Batch::with_demand(
            DomainSpan::new(0, self.row_count),
            struct_array,
            output_demand,
        );
        ctx.push(batch)?;
        ctx.seal()?;
        state.sealed = true;
        ctx.trace(format!("{}: emitted assembled struct batch", self.label));
        Ok(WorkStatus::Finished)
    }
}
