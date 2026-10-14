// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::expr::display::ExprDisplay;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// A lambda parameter, read by name while the lambda's body is written.
///
/// Building the [`Lambda`](crate::expr::Lambda) rewrites each variable of its parameters into a
/// field of the lambda's row, so a variable left over is not a parameter of any enclosing lambda.
/// It has no dtype, cannot run, and cannot be serialized.
#[derive(Clone)]
pub struct Var;

impl ScalarFnVTable for Var {
    type Options = FieldName;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.var");
        *ID
    }

    fn serialize(&self, name: &FieldName) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("variable `{name}` is not a parameter of an enclosing lambda")
    }

    fn arity(&self, _name: &FieldName) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _name: &FieldName, child_idx: usize) -> ChildName {
        unreachable!("Invalid child index {child_idx} for var()")
    }

    fn fmt_sql(
        &self,
        name: &FieldName,
        _expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{name}")
    }

    fn return_dtype(&self, name: &FieldName, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_bail!("variable `{name}` is not a parameter of an enclosing lambda")
    }

    fn execute(
        &self,
        name: &FieldName,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("variable `{name}` is not a parameter of an enclosing lambda")
    }
}
