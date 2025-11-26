// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::traversal::Node;
use crate::expr::{
    ChildName, ExecutionArgs, ExprId, Expression, ExpressionView, StatsCatalog, VTable,
};
use crate::functions::{ScalarFunctionVTable, Signature};
use crate::stats::Stat;
use crate::{functions, ArrayRef};
use itertools::Itertools;
use std::fmt::Formatter;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_vector::Vector;

/// An expression representing a call to a scalar function.
pub struct Call {
    vtable: ScalarFunctionVTable,
}

impl Call {
    pub fn new<F: functions::VTable>(vtable: F) -> Self {
        Self {
            vtable: ScalarFunctionVTable::new(vtable),
        }
    }

    pub fn from_static<F: functions::VTable>(vtable: &'static F) -> Self {
        Self {
            vtable: ScalarFunctionVTable::new_static(vtable),
        }
    }
}

/// Additional logic required to wrap a scalar function as an expression.
///
/// This trait contains logic for formatting, statistics, and other expression-specific
/// behavior that is not part of the core scalar function implementation.
pub trait ScalarFunctionVTableExt<F: functions::VTable>: 'static + Send + Sync {}

impl<F: functions::VTable> VTable for Call<F> {
    /// The instance data for a `Call` expression are the function's options.
    type Instance = F::Options;

    fn id(&self) -> ExprId {
        self.vtable.id()
    }

    fn serialize(&self, options: &F::Options) -> VortexResult<Option<Vec<u8>>> {
        self.vtable.serialize(options)
    }

    fn deserialize(&self, bytes: &[u8]) -> VortexResult<Self::Instance> {
        self.vtable.deserialize(bytes)
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        let arity = self.vtable.signature(expr.data()).arity();
        if arity != expr.children_count() {
            vortex_bail!(
                "Function '{}' expects {} arguments, but got {}",
                self.id(),
                arity,
                expr.children_count()
            );
        }
        Ok(())
    }

    fn child_name(&self, options: &F::Options, child_idx: usize) -> ChildName {
        ChildName::from(Arc::from(
            self.vtable
                .signature(options)
                .child_name(child_idx)
                .unwrap_or_else(|| "<unnamed>".to_string()),
        ))
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.id())?;
        for (i, child) in expr.children().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            child.fmt_sql(f)?;
        }

        if expr.data() != &Default::default() {
            if expr.children_count() > 0 {
                write!(f, " ")?;
            }
            write!(f, "options: {}", expr.data())?;
        }

        write!(f, ")")
    }

    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let arg_dtypes: Vec<_> = expr
            .children()
            .iter()
            .map(|c| c.return_dtype(scope))
            .try_collect()?;
        self.vtable.return_dtype(expr.data(), &arg_dtypes)
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        // TODO(ngates): we evaluate a function by wrapping it in a CallArray.
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
