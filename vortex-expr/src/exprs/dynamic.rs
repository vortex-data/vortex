// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{Operator, compare};
use vortex_array::{Array, ArrayRef, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_proto::expr as pb;
use vortex_scalar::{Scalar, ScalarValue};

use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, Scope, VTable, vtable};

vtable!(DynamicComparison);

/// A dynamic comparison expression can be used to capture a comparison to a value that can change
/// during the execution of a query, such as when a compute engine pushes down an ORDER BY + LIMIT
/// operation and is able to progressively tighten the bounds of the filter.
///
/// Hash and PartialEq are implemented based on the ptr of the value function, such that the
/// internal value doesn't impact the hash of an expression tree.
#[derive(Clone)]
pub struct DynamicComparisonExpr {
    lhs: ExprRef,
    operator: Operator,
    rhs_value: Arc<dyn Fn() -> Option<ScalarValue> + Send + Sync>,
    rhs_dtype: DType,
    // Default value for the dynamic comparison.
    default: bool,
}

impl Hash for DynamicComparisonExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.lhs.hash(state);
        self.operator.hash(state);
        Arc::as_ptr(&self.rhs_value).hash(state);
        self.rhs_dtype.hash(state);
        self.default.hash(state);
    }
}

impl PartialEq for DynamicComparisonExpr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.rhs_value, &other.rhs_value)
            && self.default == other.default
            && self.operator == other.operator
            && self.lhs.eq(&other.lhs)
            && self.rhs_dtype == other.rhs_dtype
    }
}

impl Eq for DynamicComparisonExpr {}

impl Debug for DynamicComparisonExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicComparisonExpr")
            .field("lhs", &self.lhs)
            .field("operator", &self.operator)
            .field(
                "rhs_value",
                &"<dyn Fn() -> Option<ScalarValue> + Send + Sync>",
            )
            .field("rhs_dtype", &self.rhs_dtype)
            .field("default", &self.default)
            .finish()
    }
}

pub struct DynamicComparisonExprEncoding;

impl VTable for DynamicComparisonVTable {
    type Expr = DynamicComparisonExpr;
    type Encoding = DynamicComparisonExprEncoding;
    type Metadata = ProstMetadata<pb::LiteralOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("dynamic")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(DynamicComparisonExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        None
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.lhs]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(DynamicComparisonExpr {
            lhs: children[0].clone(),
            operator: expr.operator,
            rhs_value: expr.rhs_value.clone(),
            rhs_dtype: expr.rhs_dtype.clone(),
            default: expr.default,
        })
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        vortex_bail!("DynamicComparison expression does not support building from metadata");
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        if let Some(value) = expr.scalar() {
            let lhs = expr.lhs.evaluate(scope)?;
            let rhs = ConstantArray::new(value, scope.len());
            return compare(lhs.as_ref(), rhs.as_ref(), expr.operator);
        }

        // Otherwise, we return the default value.
        let lhs = expr.return_dtype(scope.dtype())?;
        Ok(ConstantArray::new(
            Scalar::new(
                DType::Bool(lhs.nullability() | expr.rhs_dtype.nullability()),
                expr.default.into(),
            ),
            scope.len(),
        )
        .into_array())
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let lhs = expr.lhs.return_dtype(scope)?;
        if !expr.rhs_dtype.eq_ignore_nullability(&lhs) {
            vortex_bail!(
                "Incompatible dtypes for dynamic comparison: expected {} (ignore nullability) but got {}",
                &expr.rhs_dtype,
                lhs
            );
        }
        Ok(DType::Bool(
            lhs.nullability() | expr.rhs_dtype.nullability(),
        ))
    }
}

impl DynamicComparisonExpr {
    pub fn new(
        rhs: ExprRef,
        operator: Operator,
        rhs_value: impl Fn() -> Option<ScalarValue> + Send + Sync + 'static,
        rhs_dtype: DType,
        default: bool,
    ) -> Self {
        DynamicComparisonExpr {
            lhs: rhs,
            operator,
            rhs_value: Arc::new(rhs_value),
            rhs_dtype,
            default,
        }
    }

    pub fn scalar(&self) -> Option<Scalar> {
        (self.rhs_value)().map(|v| Scalar::new(self.rhs_dtype.clone(), v))
    }
}

impl Display for DynamicComparisonExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dynamic({}) {} {}",
            &self.lhs, self.operator, &self.rhs_dtype,
        )
    }
}

impl AnalysisExpr for DynamicComparisonExpr {}
