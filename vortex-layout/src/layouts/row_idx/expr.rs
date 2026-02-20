// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Arity;
use vortex_array::expr::ChildName;
use vortex_array::expr::EmptyOptions;
use vortex_array::expr::ExecutionArgs;
use vortex_array::expr::ExprId;
use vortex_array::expr::Expression;
use vortex_array::expr::VTable;
use vortex_array::expr::VTableExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

pub struct RowIdx;

impl VTable for RowIdx {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.row_idx")
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

    fn execute(&self, _options: &Self::Options, _args: ExecutionArgs) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "RowIdxExpr should not be executed directly, use it in the context of a Vortex scan and it will be substituted for a row index array"
        );
    }
}

pub fn row_idx() -> Expression {
    RowIdx.new_expr(EmptyOptions, [])
}
