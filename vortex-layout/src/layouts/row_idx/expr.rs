// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::expr::ChildName;
use vortex_array::expr::ExprId;
use vortex_array::expr::Expression;
use vortex_array::expr::ExpressionView;
use vortex_array::expr::VTable;
use vortex_array::expr::VTableExt;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::vortex_bail;
use vortex_error::VortexResult;

pub struct RowIdx;

impl VTable for RowIdx {
    type Options = ();

    fn id(&self) -> ExprId {
        ExprId::from("vortex.row_idx")
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if !expr.children().is_empty() {
            vortex_bail!(
                "RowIdx expression does not have children, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!()
    }

    fn fmt_sql(&self, _expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "#row_idx")
    }

    fn return_dtype(&self, _expr: &ExpressionView<Self>, _scope: &DType) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn evaluate(&self, _expr: &ExpressionView<Self>, _scope: &ArrayRef) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "RowIdxExpr should not be evaluated directly, use it in the context of a Vortex scan and it will be substituted for a row index array"
        );
    }
}

pub fn row_idx() -> Expression {
    RowIdx.new_expr((), [])
}
