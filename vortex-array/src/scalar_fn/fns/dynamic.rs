// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeVisitor;
use crate::expr::traversal::TraversalOrder;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

/// A dynamic comparison expression can be used to capture a comparison to a value that can change
/// during the execution of a query, such as when a compute engine pushes down an ORDER BY + LIMIT
/// operation and is able to progressively tighten the bounds of the filter.
#[derive(Clone)]
pub struct DynamicComparison;

impl ScalarFnVTable for DynamicComparison {
    type Options = DynamicComparisonExpr;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.dynamic")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            _ => unreachable!(),
        }
    }

    fn fmt_sql(
        &self,
        dynamic: &DynamicComparisonExpr,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        expr.child(0).fmt_sql(f)?;
        write!(f, " {} dynamic(", dynamic.operator)?;
        match dynamic.scalar() {
            None => write!(f, "scalar=<none>")?,
            Some(scalar) => write!(f, "scalar={scalar}")?,
        }
        write!(f, ")")
    }

    fn return_dtype(
        &self,
        dynamic: &DynamicComparisonExpr,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        if !dynamic.rhs.dtype.eq_ignore_nullability(lhs) {
            vortex_bail!(
                "Incompatible dtypes for dynamic comparison: expected {} (ignore nullability) but got {}",
                &dynamic.rhs.dtype,
                lhs
            );
        }
        Ok(DType::Bool(
            lhs.nullability() | dynamic.rhs.dtype.nullability(),
        ))
    }

    fn execute(
        &self,
        data: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(scalar) = data.rhs.scalar() {
            let lhs = args.get(0)?;
            let rhs = ConstantArray::new(scalar, args.row_count()).into_array();

            let delegate_args = VecExecutionArgs::new(vec![lhs, rhs], args.row_count());
            return Binary
                .bind(Operator::from(data.operator))
                .execute(&delegate_args, ctx);
        }
        let ret_dtype =
            DType::Bool(args.get(0)?.dtype().nullability() | data.rhs.dtype.nullability());

        Ok(ConstantArray::new(
            Scalar::try_new(ret_dtype, Some(data.default.into()))?,
            args.row_count(),
        )
        .into_array())
    }

    fn stat_falsification(
        &self,
        dynamic: &DynamicComparisonExpr,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        let lhs = expr.child(0);
        match dynamic.operator {
            CompareOperator::Eq | CompareOperator::NotEq => None,
            CompareOperator::Gt => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: CompareOperator::Lte,
                    rhs: Arc::clone(&dynamic.rhs),
                    default: !dynamic.default,
                },
                vec![lhs.stat_max(catalog)?],
            )),
            CompareOperator::Gte => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: CompareOperator::Lt,
                    rhs: Arc::clone(&dynamic.rhs),
                    default: !dynamic.default,
                },
                vec![lhs.stat_max(catalog)?],
            )),
            CompareOperator::Lt => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: CompareOperator::Gte,
                    rhs: Arc::clone(&dynamic.rhs),
                    default: !dynamic.default,
                },
                vec![lhs.stat_min(catalog)?],
            )),
            CompareOperator::Lte => Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: CompareOperator::Gt,
                    rhs: Arc::clone(&dynamic.rhs),
                    default: !dynamic.default,
                },
                vec![lhs.stat_min(catalog)?],
            )),
        }
    }

    // Defer to the child
    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
pub struct DynamicComparisonExpr {
    pub(crate) operator: CompareOperator,
    pub(crate) rhs: Arc<Rhs>,
    // Default value for the dynamic comparison.
    pub(crate) default: bool,
}

impl DynamicComparisonExpr {
    pub fn scalar(&self) -> Option<Scalar> {
        (self.rhs.value)().map(|v| {
            Scalar::try_new(self.rhs.dtype.clone(), Some(v))
                .vortex_expect("`DynamicComparisonExpr` was invalid")
        })
    }
}

impl Display for DynamicComparisonExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            self.operator,
            self.scalar()
                .map_or_else(|| "<none>".to_string(), |v| v.to_string())
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
pub(crate) struct Rhs {
    // The right-hand side value is a function that returns an `Option<ScalarValue>`.
    pub(crate) value: Arc<dyn Fn() -> Option<ScalarValue> + Send + Sync>,
    // The data type of the right-hand side value.
    pub(crate) dtype: DType,
}

impl Rhs {
    pub fn scalar(&self) -> Option<Scalar> {
        (self.value)().map(|v| {
            Scalar::try_new(self.dtype.clone(), Some(v)).vortex_expect("`Rhs` was invalid")
        })
    }
}

impl Debug for Rhs {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rhs")
            .field("value", &"<dyn Fn() -> Option<ScalarValue> + Send + Sync>")
            .field("dtype", &self.dtype)
            .finish()
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
            .map(|expr| {
                (expr.rhs.value)().map(|v| {
                    Scalar::try_new(expr.rhs.dtype.clone(), Some(v))
                        .vortex_expect("`DynamicExprUpdates` was invalid")
                })
            })
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicI32;
    use std::sync::atomic::Ordering;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::dynamic;
    use crate::expr::root;
    #[test]
    fn return_dtype_bool() -> VortexResult<()> {
        let expr = dynamic(
            CompareOperator::Lt,
            || Some(5i32.into()),
            DType::Primitive(PType::I32, Nullability::NonNullable),
            true,
            root(),
        );
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        assert_eq!(
            expr.return_dtype(&input_dtype)?,
            DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn execute_with_value() -> VortexResult<()> {
        let input = buffer![1i32, 5, 10].into_array();
        let expr = dynamic(
            CompareOperator::Lt,
            || Some(5i32.into()),
            DType::Primitive(PType::I32, Nullability::NonNullable),
            true,
            root(),
        );
        let result = input.apply(&expr)?;
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]));
        Ok(())
    }

    #[test]
    fn execute_without_value_default_true() -> VortexResult<()> {
        let input = buffer![1i32, 5, 10].into_array();
        let expr = dynamic(
            CompareOperator::Lt,
            || None,
            DType::Primitive(PType::I32, Nullability::NonNullable),
            true,
            root(),
        );
        let result = input.apply(&expr)?;
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, true]));
        Ok(())
    }

    #[test]
    fn execute_without_value_default_false() -> VortexResult<()> {
        let input = buffer![1i32, 5, 10].into_array();
        let expr = dynamic(
            CompareOperator::Lt,
            || None,
            DType::Primitive(PType::I32, Nullability::NonNullable),
            false,
            root(),
        );
        let result = input.apply(&expr)?;
        assert_arrays_eq!(result, BoolArray::from_iter([false, false, false]));
        Ok(())
    }

    #[test]
    fn execute_value_flips() -> VortexResult<()> {
        let threshold = Arc::new(AtomicI32::new(5));
        let threshold_clone = Arc::clone(&threshold);
        let expr = dynamic(
            CompareOperator::Lt,
            move || Some(threshold_clone.load(Ordering::SeqCst).into()),
            DType::Primitive(PType::I32, Nullability::NonNullable),
            true,
            root(),
        );
        let input = buffer![1i32, 5, 10].into_array();

        let result = input.clone().apply(&expr)?;
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]));

        threshold.store(10, Ordering::SeqCst);
        let result = input.apply(&expr)?;
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, false]));

        Ok(())
    }
}
