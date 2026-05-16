//! `Operator` trait and `PhysicalPlan`.

use crate::Domain;
use crate::OutputContract;
use crate::physical_plan::abi::SinkNode;
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::error::PlanValidationError;
use crate::physical_plan::lowering::LoweringCtx;
use crate::physical_plan::lowering::PipelineTail;

/// Plan-time operator. Each operator implements one method: `lower`.
///
/// `Send + Sync` is required so that operators can be moved into
/// `Box<dyn Operator>` slots stored on `Send`/`Sync` types (e.g.
/// `Gather` holds its child operator subtrees in a `Mutex<...>` on a
/// `SourceNode`, which is itself `Send + Sync`).
pub trait Operator: Send + Sync {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()>;
}

pub struct PhysicalPlan {
    output: OutputContract,
    root: Box<dyn Operator>,
}

impl PhysicalPlan {
    pub fn new(output: OutputContract, root: Box<dyn Operator>) -> Self {
        Self { output, root }
    }

    pub fn output(&self) -> &OutputContract {
        &self.output
    }

    pub fn root(&self) -> &dyn Operator {
        self.root.as_ref()
    }

    pub fn lower<E>(
        &self,
        ctx: &mut dyn LoweringCtx,
        output_domain: Domain,
        sink: E,
    ) -> BuildResult<()>
    where
        E: SinkNode,
    {
        ctx.register_domain(output_domain.clone())?;
        self.root.lower(
            ctx,
            PipelineTail::new(output_domain, self.output.clone(), sink),
        )
    }

    pub fn validate(&self) -> Result<(), PlanValidationError> {
        Ok(())
    }
}
