// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{Operator, compare};
use vortex_array::{Array, ArrayRef, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_proto::expr as pb;
use vortex_scalar::{Scalar, ScalarValue};

use crate::traversal::{Node, NodeVisitor, TraversalOrder};
use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, Scope, VTable, vtable};

vtable!(DynamicComparison);

/// A dynamic comparison expression can be used to capture a comparison to a value that can change
/// during the execution of a query, such as when a compute engine pushes down an ORDER BY + LIMIT
/// operation and is able to progressively tighten the bounds of the filter.
#[derive(Clone, Debug, Hash)]
pub struct DynamicComparisonExpr {
    lhs: ExprRef,
    operator: Operator,
    rhs: Arc<Rhs>,
    // Default value for the dynamic comparison.
    default: bool,
}

impl PartialEq for DynamicComparisonExpr {
    fn eq(&self, other: &Self) -> bool {
        self.lhs.eq(&other.lhs)
            && self.operator == other.operator
            && self.rhs == other.rhs
            && self.default == other.default
    }
}
impl Eq for DynamicComparisonExpr {}

/// Hash and PartialEq are implemented based on the ptr of the value function, such that the
/// internal value doesn't impact the hash of an expression tree.
struct Rhs {
    // The right-hand side value is a function that returns an `Option<ScalarValue>`.
    value: Arc<dyn Fn() -> Option<ScalarValue> + Send + Sync>,
    // The data type of the right-hand side value.
    dtype: DType,
    // Tracks how many times the value has changed. Note that this may over-estimate the changes
    // in order to avoid lock contention.
    version: AtomicUsize,
    // The previous value of the right-hand side, used to detect changes.
    previous_value: Option<Scalar>,
}

impl Hash for Rhs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.value).hash(state);
    }
}

impl PartialEq for Rhs {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
}
impl Eq for Rhs {}

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
            "dynamic({}) {} {}",
            &self.lhs, self.operator, &self.rhs.dtype,
        )
    }
}

impl AnalysisExpr for DynamicComparisonExpr {}

/// A utility for checking whether any dynamic expressions have been updated.
pub struct DynamicExprHash(Box<[DynamicComparisonExpr]>);

impl DynamicExprHash {
    pub fn new(expr: &ExprRef) -> Self {
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

        DynamicExprHash(visitor.0.into_boxed_slice())
    }

    pub fn checksum(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl Hash for DynamicExprHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for expr in &self.0 {
            expr.scalar().hash(state);
        }
    }
}
