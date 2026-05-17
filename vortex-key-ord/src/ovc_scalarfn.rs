// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `Ovc`: OVC as a real Vortex [`ScalarFnVTable`].
//!
//! Users write `ovc(input_expr)` and the executor dispatches per-encoding
//! through the runtime [`ArrayKernels`] registry.
//!
//! [`ArrayKernels`]: vortex_array::optimizer::kernels::ArrayKernels

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::SessionExt;

use crate::stream_kernel::OvcKernel;
use crate::stream_kernel::dispatch_ovc_encode;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Primitive;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::optimizer::kernels::ArrayKernels;
use vortex_array::optimizer::kernels::ExecuteParentFn;
use vortex_array::optimizer::kernels::ReduceParentFn;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;

/// OVC scalar function. Arity 1; output dtype matches input.
#[derive(Clone, Debug, Default)]
pub struct Ovc;

impl Display for Ovc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "vortex.ovc")
    }
}

impl ScalarFnVTable for Ovc {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.ovc")
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Ovc has arity 1"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(arg_dtypes[0].clone())
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(dispatch_ovc_encode(&args.get(0)?, 0))
    }

    fn is_null_sensitive(&self, _: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _: &Self::Options) -> bool {
        false
    }
}

fn ovc_execute<V: OvcKernel>(
    child: &ArrayRef,
    _parent: &ArrayRef,
    _child_idx: usize,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let view = child
        .as_typed::<V>()
        .ok_or_else(|| vortex_err!("ovc_execute: encoding mismatch"))?;
    Ok(Some(<V as OvcKernel>::ovc_encode(view, 0)))
}

/// Pre-empts `ChunkedUnaryScalarFnPushDownRule` so the whole-array
/// `OvcKernel for Chunked` runs instead of the per-chunk push-down.
fn ovc_chunked_reduce_parent(
    child: &ArrayRef,
    _parent: &ArrayRef,
    _child_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    let view = child
        .as_typed::<Chunked>()
        .ok_or_else(|| vortex_err!("not chunked"))?;
    Ok(Some(<Chunked as OvcKernel>::ovc_encode(view, 0)))
}

/// Register [`Ovc`] and per-encoding kernels into `session`. Idempotent
/// at the registry level (ArcSwap); call once at session init.
pub fn register_ovc(session: &impl SessionExt) {
    session.scalar_fns().register(Ovc);

    let kernels = session.get::<ArrayKernels>();
    let ovc_id = <Ovc as ScalarFnVTable>::id(&Ovc);

    for (encoding_id, f) in [
        (
            <Primitive as VTable>::id(&Primitive),
            ovc_execute::<Primitive> as ExecuteParentFn,
        ),
        (
            <Constant as VTable>::id(&Constant),
            ovc_execute::<Constant> as ExecuteParentFn,
        ),
        (
            <Chunked as VTable>::id(&Chunked),
            ovc_execute::<Chunked> as ExecuteParentFn,
        ),
    ] {
        kernels.register_execute_parent(ovc_id, encoding_id, &[f]);
    }

    kernels.register_reduce_parent(
        ovc_id,
        <Chunked as VTable>::id(&Chunked),
        &[ovc_chunked_reduce_parent as ReduceParentFn],
    );
}

/// `ovc(input)` expression builder.
pub fn ovc(input: Expression) -> Expression {
    Ovc.new_expr(EmptyOptions, [input])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::expr::root;

    #[test]
    fn end_to_end_constant_preserves_encoding() -> VortexResult<()> {
        register_ovc(&*LEGACY_SESSION);
        let input = ConstantArray::new(7u64, 128).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = input
            .apply(&ovc(root()))?
            .execute::<ArrayRef>(&mut ctx)?;
        assert_eq!(result.len(), 128);
        assert!(
            result.as_opt::<Constant>().is_some(),
            "expected ConstantArray, got {}",
            result.encoding_id()
        );
        Ok(())
    }

    #[test]
    fn end_to_end_chunked() -> VortexResult<()> {
        use vortex_array::arrays::ChunkedArray;
        use vortex_array::dtype::Nullability;
        use vortex_array::dtype::PType;

        register_ovc(&*LEGACY_SESSION);
        let chunks: Vec<ArrayRef> = (0..5)
            .map(|_| ConstantArray::new(13u64, 20).into_array())
            .collect();
        let chunked = ChunkedArray::try_new(
            chunks,
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result: ArrayRef = chunked
            .apply(&ovc(root()))?
            .execute::<ArrayRef>(&mut ctx)?;
        assert_eq!(result.len(), 100);
        Ok(())
    }
}
