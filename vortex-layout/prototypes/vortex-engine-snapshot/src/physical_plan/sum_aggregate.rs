//! `SumCountAggregate`: simple all-rows aggregator that emits one
//! row at end-of-input carrying `(sum, count)` packed as two
//! consecutive i64 values.
//!
//! Used to compute averages: caller divides `sum / count` after
//! draining the sink.
//!
//! Implements `TransformNode`. Push pattern: ingest every input
//! batch, accumulate into local state. After `finish_input`,
//! `poll_next_output` returns one final batch holding `[sum,
//! count]`, then `Finished`.

use std::task::Poll;

use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;

use crate::Domain;
use crate::DomainSpan;
use crate::EngineError;
use crate::EngineResult;
use crate::OutputContract;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, TransformCtx, TransformNode, TransformOutput,
};
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::lowering::{LoweringCtx, PipelineTail};
use crate::physical_plan::plan::Operator;

pub struct SumCountAggregate {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    output_domain: Domain,
    output_contract: OutputContract,
    input: Box<dyn Operator>,
}

impl SumCountAggregate {
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        input_contract: OutputContract,
        output_domain: Domain,
        output_contract: OutputContract,
        input: Box<dyn Operator>,
    ) -> Self {
        Self {
            label: label.into(),
            input_domain,
            input_contract,
            output_domain,
            output_contract,
            input,
        }
    }
}

pub struct SumCountNode {
    label: String,
}

#[derive(Default)]
pub struct SumCountLocal {
    sum: i128,
    count: i64,
    input_done: bool,
    emitted: bool,
}

impl TransformNode for SumCountNode {
    type LocalState = SumCountLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(SumCountLocal::default())
    }

    fn can_accept_input(&self, local: &Self::LocalState) -> bool {
        !local.input_done
    }

    fn push_input(
        &self,
        local: &mut Self::LocalState,
        batch: Batch,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        // Materialise the input array as a canonical PrimitiveArray
        // of i64. Sum the buffer, count non-null entries.
        let array = batch.array().clone();
        #[expect(deprecated)]
        let canonical = array
            .to_canonical()
            .map_err(|e| EngineError::message(format!("canonicalize: {e}")))?;
        let primitive: PrimitiveArray = match canonical {
            Canonical::Primitive(p) => p,
            other => other
                .into_array()
                .try_downcast::<Primitive>()
                .map_err(|_| EngineError::message("sum aggregate expected primitive"))?,
        };
        if primitive.ptype() != PType::I64 {
            return Err(EngineError::message(format!(
                "sum aggregate expected i64, got {:?}",
                primitive.ptype()
            )));
        }
        let buffer = primitive.to_buffer::<i64>();
        let validity = primitive
            .validity()
            .map_err(|e| EngineError::message(format!("validity: {e}")))?;
        for (idx, value) in buffer.iter().enumerate() {
            let is_valid = match &validity {
                vortex_array::validity::Validity::NonNullable
                | vortex_array::validity::Validity::AllValid => true,
                vortex_array::validity::Validity::AllInvalid => false,
                vortex_array::validity::Validity::Array(_) => {
                    validity.is_valid(idx).unwrap_or(false)
                }
            };
            if is_valid {
                local.sum = local.sum.wrapping_add(*value as i128);
                local.count += 1;
            }
        }
        Ok(())
    }

    fn finish_input(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        local.input_done = true;
        Ok(())
    }

    fn poll_next_output(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput> {
        if local.emitted {
            return Poll::Ready(Ok(TransformOutput::Finished));
        }
        if !local.input_done {
            return Poll::Ready(Ok(TransformOutput::NeedInput));
        }
        // Emit one batch carrying [sum_low_i64, sum_high_i64, count].
        // The i128 sum is split into two i64 halves so we can
        // represent the full range without overflow.
        let unsigned = local.sum as u128;
        let sum_low = (unsigned & 0xFFFF_FFFF_FFFF_FFFF) as i64;
        let sum_high = ((unsigned >> 64) & 0xFFFF_FFFF_FFFF_FFFF) as i64;
        let span = DomainSpan::new(0, 3);
        let array =
            PrimitiveArray::from_iter(vec![sum_low, sum_high, local.count]).into_array();
        local.emitted = true;
        Poll::Ready(Ok(TransformOutput::Batch(Batch::new(array, span))))
    }
}

impl Operator for SumCountAggregate {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        // SumCountAggregate is shape-changing: it changes the
        // pipeline's domain from input_domain to output_domain.
        // The tail's expected_input_domain is our *output* domain;
        // we register both and prepend the transform that operates
        // on the input_domain side.
        ctx.register_domain(self.output_domain.clone())?;
        let tail = tail.prepend_transform(
            self.input_domain.clone(),
            self.input_contract.clone(),
            SumCountNode {
                label: self.label.clone(),
            },
        );
        self.input.lower(ctx, tail)
    }
}
