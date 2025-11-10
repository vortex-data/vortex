// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::ConstantArray;
use crate::compute::{Operator, compare};
use crate::expr::traversal::{NodeExt, NodeVisitor, TraversalOrder};
use crate::expr::{ChildName, ExprId, Expression, ExpressionView, StatsCatalog, VTable, VTableExt};
use crate::{Array, ArrayRef, IntoArray};

/// A dynamic comparison expression can be used to capture a comparison to a value that can change
/// during the execution of a query, such as when a compute engine pushes down an ORDER BY + LIMIT
/// operation and is able to progressively tighten the bounds of the filter.
pub struct DynamicComparison;

impl VTable for DynamicComparison {
    type Instance = DynamicComparisonExpr;

    fn id(&self) -> ExprId {
        ExprId::new_ref("vortex.dynamic")
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 1 {
            vortex_bail!(
                "DynamicComparison expression requires exactly one child, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            _ => unreachable!(),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        expr.lhs().fmt_sql(f)?;
        write!(f, " {} dynamic(", expr.data())?;
        match expr.scalar() {
            None => write!(f, "<none>")?,
            Some(scalar) => write!(f, "{}", scalar)?,
        }
        write!(f, ")")
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let lhs = expr.lhs().return_dtype(scope)?;
        if !expr.data().rhs.dtype.eq_ignore_nullability(&lhs) {
            vortex_bail!(
                "Incompatible dtypes for dynamic comparison: expected {} (ignore nullability) but got {}",
                &expr.data().rhs.dtype,
                lhs
            );
        }
        Ok(DType::Bool(
            lhs.nullability() | expr.data().rhs.dtype.nullability(),
        ))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        if let Some(value) = expr.scalar() {
            let lhs = expr.lhs().evaluate(scope)?;
            let rhs = ConstantArray::new(value, scope.len());
            return compare(lhs.as_ref(), rhs.as_ref(), expr.data().operator);
        }

        // Otherwise, we return the default value.
        let lhs = expr.return_dtype(scope.dtype())?;
        Ok(ConstantArray::new(
            Scalar::new(
                DType::Bool(lhs.nullability() | expr.data().rhs.dtype.nullability()),
                expr.data().default.into(),
            ),
            scope.len(),
        )
        .into_array())
    }

    fn stat_falsification(
        &self,
        expr: &ExpressionView<DynamicComparison>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        match expr.data().operator {
            Operator::Gt => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: Operator::Lte,
                    rhs: expr.data().rhs.clone(),
                    default: !expr.data().default,
                },
                vec![expr.lhs().stat_max(catalog)?],
            )),
            Operator::Gte => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: Operator::Lt,
                    rhs: expr.data().rhs.clone(),
                    default: !expr.data().default,
                },
                vec![expr.lhs().stat_max(catalog)?],
            )),
            Operator::Lt => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: Operator::Gte,
                    rhs: expr.data().rhs.clone(),
                    default: !expr.data().default,
                },
                vec![expr.lhs().stat_min(catalog)?],
            )),
            Operator::Lte => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: Operator::Gt,
                    rhs: expr.data().rhs.clone(),
                    default: !expr.data().default,
                },
                vec![expr.lhs().stat_min(catalog)?],
            )),
            _ => None,
        }
    }
}

pub fn dynamic(
    operator: Operator,
    rhs_value: impl Fn() -> Option<ScalarValue> + Send + Sync + 'static,
    rhs_dtype: DType,
    default: bool,
    lhs: Expression,
) -> Expression {
    DynamicComparison.new_expr(
        DynamicComparisonExpr {
            operator,
            rhs: Arc::new(Rhs {
                value: Arc::new(rhs_value),
                dtype: rhs_dtype,
            }),
            default,
        },
        [lhs],
    )
}

#[derive(Clone, Debug)]
pub struct DynamicComparisonExpr {
    operator: Operator,
    rhs: Arc<Rhs>,
    // Default value for the dynamic comparison.
    default: bool,
}

impl DynamicComparisonExpr {
    pub fn scalar(&self) -> Option<Scalar> {
        (self.rhs.value)().map(|v| Scalar::new(self.rhs.dtype.clone(), v))
    }
}

impl Display for DynamicComparisonExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            self.operator,
            self.scalar()
                .map_or("<none>".to_string(), |v| v.to_string())
        )
    }
}

impl PartialEq for DynamicComparisonExpr {
    fn eq(&self, other: &Self) -> bool {
        self.operator == other.operator
            && Arc::ptr_eq(&self.rhs, &other.rhs)
            && self.default == other.default
    }
}
impl Eq for DynamicComparisonExpr {}

impl Hash for DynamicComparisonExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operator.hash(state);
        Arc::as_ptr(&self.rhs).hash(state);
        self.default.hash(state);
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rhs")
            .field("value", &"<dyn Fn() -> Option<ScalarValue> + Send + Sync>")
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl ExpressionView<'_, DynamicComparison> {
    pub fn lhs(&self) -> &Expression {
        &self.children()[0]
    }

    pub fn scalar(&self) -> Option<Scalar> {
        (self.data().rhs.value)().map(|v| Scalar::new(self.data().rhs.dtype.clone(), v))
    }
}

/// A utility for checking whether any dynamic expressions have been updated.
pub struct DynamicExprUpdates {
    exprs: Box<[DynamicComparisonExpr]>,
    // Track the latest observed versions of each dynamic expression, along with a version counter.
    prev_versions: Mutex<(u64, Vec<Option<Scalar>>)>,
}

impl DynamicExprUpdates {
    pub fn new(expr: &Expression) -> Option<Self> {
        #[derive(Default)]
        struct Visitor(Vec<DynamicComparisonExpr>);

        impl NodeVisitor<'_> for Visitor {
            type NodeTy = Expression;

            fn visit_down(&mut self, node: &'_ Self::NodeTy) -> VortexResult<TraversalOrder> {
                if let Some(dynamic) = node.as_opt::<DynamicComparison>() {
                    self.0.push(dynamic.data().clone());
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
