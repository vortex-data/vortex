// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

#[derive(Clone)]
pub struct RowIdx;

impl ScalarFnVTable for RowIdx {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.row_idx")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _instance: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!()
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "#row_idx")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "RowIdxExpr should not be executed directly, use it in the context of a Vortex scan and it will be substituted for a row index array"
        );
    }
}

pub fn row_idx() -> Expression {
    RowIdx.new_expr(EmptyOptions, [])
}
