// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::{
    ChildName, ExecutionArgs, ExprId, Expression, ExpressionView, StatsCatalog, VTable,
};
use crate::functions::ScalarFunction;
use crate::stats::Stat;
use crate::ArrayRef;
use std::fmt::Formatter;
use std::marker::PhantomData;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Vector;

/// An expression representing a call to a scalar function.
pub struct Call<F: ScalarFunction> {
    _phantom: PhantomData<F>,
}

impl<F: ScalarFunction> VTable for Call<F> {
    type Instance = F;

    fn id(&self) -> ExprId {
        todo!()
    }

    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        todo!()
    }

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        todo!()
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        todo!()
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        todo!()
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }

    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        todo!()
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        todo!()
    }

    fn execute(&self, _data: &Self::Instance, _args: ExecutionArgs) -> VortexResult<Vector> {
        todo!()
    }

    fn stat_falsification(
        &self,
        _expr: &ExpressionView<Self>,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        todo!()
    }

    fn stat_expression(
        &self,
        _expr: &ExpressionView<Self>,
        _stat: Stat,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        todo!()
    }
}
