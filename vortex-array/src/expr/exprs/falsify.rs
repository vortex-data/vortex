// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ConstantArray;
use crate::expr::{ChildName, ExprId, ExpressionView, VTable};
use crate::{Array, ArrayRef, IntoArray};
use std::fmt::Formatter;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};
use vortex_vector::bool::BoolVectorMut;
use vortex_vector::{Vector, VectorMutOps, VectorOps};

/// Falsify is an expression that returns `true` if the given predicate is provably false, and
/// `false` otherwise.
///
/// It has limited use in an execution context and is primarily intended for use by other
/// expressions to perform falsification push-down, ultimately allowing arrays and layouts to
/// reduce this expression into a stats-based falsification predicate.
///
/// The built-in optimization rules implement de-Morgan's laws to push FALSIFY down through AND,
/// OR, and NOT.
struct Falsify;

impl VTable for Falsify {
    type Instance = ();

    fn id(&self) -> ExprId {
        ExprId::from("vortex.falsify")
    }

    fn validate(&self, _expr: &ExpressionView<Self>) -> VortexResult<()> {
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, _child_idx: usize) -> ChildName {
        ChildName::from("predicate")
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FALSIFY(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let predicate = expr.return_dtype(scope)?;
        if !predicate.is_boolean() {
            vortex_bail!("FALSIFY predicate must be boolean");
        }

        // Regardless of the predicate nullability, FALSIFY always returns non-nullable boolean
        // TODO(ngates): does falsify require a non-nullable predicate?
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn evaluate(&self, _expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        // If the expression ends up being evaluated, it is because we were unable to falsify it
        // using other means (e.g., stats). Therefore, we return an array of `false` values.
        Ok(ConstantArray::new(false, scope.len()).into_array())
    }

    fn execute(
        &self,
        _expr: &ExpressionView<Self>,
        vector: &Vector,
        _dtype: &DType,
    ) -> VortexResult<Vector> {
        // If the expression ends up being executed, it is because we were unable to falsify it
        // using other means (e.g., stats). Therefore, we return a vector of `false` values.
        let mut bools = BoolVectorMut::with_capacity(vector.len());
        bools.append_values(false, vector.len());
        Ok(bools.freeze().into())
    }
}
