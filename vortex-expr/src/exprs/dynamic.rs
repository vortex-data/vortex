// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{Operator, compare};
use vortex_array::{Array, ArrayRef, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_proto::expr as pb;
use vortex_scalar::{Scalar, ScalarValue};

use crate::traversal::{NodeExt, NodeVisitor, TraversalOrder};
use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, StatsCatalog, VTable, vtable,
};

vtable!(DynamicComparison);

/// A dynamic comparison expression can be used to capture a comparison to a value that can change
/// during the execution of a query, such as when a compute engine pushes down an ORDER BY + LIMIT
/// operation and is able to progressively tighten the bounds of the filter.
#[derive(Clone, Debug)]
pub struct DynamicComparisonExpr {
    lhs: ExprRef,
    operator: Operator,
    rhs: Arc<Rhs>,
    // Default value for the dynamic comparison.
    default: bool,
}

impl PartialEq for DynamicComparisonExpr {
    fn eq(&self, other: &Self) -> bool {
        self.default == other.default
            && self.operator == other.operator
            && self.lhs.eq(&other.lhs)
            && Arc::ptr_eq(&self.rhs.value, &other.rhs.value)
            && self.rhs.dtype == other.rhs.dtype
    }
}
impl Eq for DynamicComparisonExpr {}

impl Hash for DynamicComparisonExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.default.hash(state);
        self.operator.hash(state);
        self.lhs.hash(state);
        Arc::as_ptr(&self.rhs.value).hash(state);
        self.rhs.dtype.hash(state);
    }
}

/// Hash and PartialEq are implemented based on the ptr of the value function, such that the
/// internal value doesn't impact the hash of an expression tree.
struct Rhs {
    // The right-hand side value is a function that returns an `Option<ScalarValue>`.
    value: Arc<dyn Fn() -> Option<ScalarValue> + Send + Sync>,
    // The data type of the right-hand side value.
    dtype: DType,
}

impl Debug for Rhs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rhs")
            .field("value", &"<dyn Fn() -> Option<ScalarValue> + Send + Sync>")
            .field("dtype", &self.dtype)
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
            rhs: expr.rhs.clone(),
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
                DType::Bool(lhs.nullability() | expr.rhs.dtype.nullability()),
                expr.default.into(),
            ),
            scope.len(),
        )
        .into_array())
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let lhs = expr.lhs.return_dtype(scope)?;
        if !expr.rhs.dtype.eq_ignore_nullability(&lhs) {
            vortex_bail!(
                "Incompatible dtypes for dynamic comparison: expected {} (ignore nullability) but got {}",
                &expr.rhs.dtype,
                lhs
            );
        }
        Ok(DType::Bool(
            lhs.nullability() | expr.rhs.dtype.nullability(),
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
            rhs: Arc::new(Rhs {
                value: Arc::new(rhs_value),
                dtype: rhs_dtype,
            }),
            default,
        }
    }

    pub fn scalar(&self) -> Option<Scalar> {
        (self.rhs.value)().map(|v| Scalar::new(self.rhs.dtype.clone(), v))
    }
}

impl Display for DynamicComparisonExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} dynamic({})",
            &self.lhs, self.operator, &self.rhs.dtype,
        )
    }
}

impl AnalysisExpr for DynamicComparisonExpr {
    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        match self.operator {
            Operator::Gt => Some(
                DynamicComparisonExpr {
                    lhs: self.lhs.max(catalog)?,
                    operator: Operator::Lte,
                    rhs: self.rhs.clone(),
                    default: !self.default,
                }
                .into_expr(),
            ),
            Operator::Gte => Some(
                DynamicComparisonExpr {
                    lhs: self.lhs.max(catalog)?,
                    operator: Operator::Lt,
                    rhs: self.rhs.clone(),
                    default: !self.default,
                }
                .into_expr(),
            ),
            Operator::Lt => Some(
                DynamicComparisonExpr {
                    lhs: self.lhs.min(catalog)?,
                    operator: Operator::Gte,
                    rhs: self.rhs.clone(),
                    default: !self.default,
                }
                .into_expr(),
            ),
            Operator::Lte => Some(
                DynamicComparisonExpr {
                    lhs: self.lhs.min(catalog)?,
                    operator: Operator::Gt,
                    rhs: self.rhs.clone(),
                    default: !self.default,
                }
                .into_expr(),
            ),
            _ => None,
        }
    }
}

/// A utility for checking whether any dynamic expressions have been updated.
pub struct DynamicExprUpdates {
    exprs: Box<[DynamicComparisonExpr]>,
    // Track the latest observed versions of each dynamic expression, along with a version counter.
    prev_versions: Mutex<(u64, Vec<Option<Scalar>>)>,
}

impl DynamicExprUpdates {
    pub fn new(expr: &ExprRef) -> Option<Self> {
        #[derive(Default)]
        struct Visitor(Vec<DynamicComparisonExpr>);

        impl NodeVisitor<'_> for Visitor {
            type NodeTy = ExprRef;

            fn visit_down(&mut self, node: &'_ Self::NodeTy) -> VortexResult<TraversalOrder> {
                if let Some(dynamic) = node.as_opt::<DynamicComparisonVTable>() {
                    self.0.push(dynamic.clone());
                }
                Ok(TraversalOrder::Continue)
            }
        }

        let mut visitor = Visitor::default();
        expr.accept(&mut visitor).vortex_expect("Infallible");

        if visitor.0.is_empty() {
            return None;
        }

        let exprs = visitor.0.into_boxed_slice();
        let prev_versions = exprs
            .iter()
            .map(|expr| (expr.rhs.value)().map(|v| Scalar::new(expr.rhs.dtype.clone(), v)))
            .collect();

        Some(Self {
            exprs,
            prev_versions: Mutex::new((0, prev_versions)),
        })
    }

    pub fn version(&self) -> u64 {
        let mut guard = self.prev_versions.lock();

        let mut updated = false;
        for (i, expr) in self.exprs.iter().enumerate() {
            let current = expr.scalar();
            if current != guard.1[i] {
                // At least one expression has been updated.
                // We don't bail out early in order to avoid false positives for future calls
                // to `is_updated`.
                updated = true;
                guard.1[i] = current;
            }
        }

        if updated {
            guard.0 += 1;
        }

        guard.0
    }
}
