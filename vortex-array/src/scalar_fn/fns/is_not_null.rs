// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::display::ExprDisplay;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::is_not_null_node;
use crate::validity::Validity;

/// In array context, reduce
///
/// IsNull(x) -> if !x.nullable lit(false) else not(x.validity())
/// IsNotNull(x) -> if !x.nullable lit(true) or x.validity()
///
/// In expression and array contexts for x, y where x is nullable but y is not,
/// reduce
///
/// 1. IsNull(and(x, y)) -> and(IsNull(x), y)
/// 2. IsNull(or(x, y)) -> and(IsNull(x), not(y))
/// 3. IsNotNull(and(x, y)) -> or(IsNotNull(x), not(y))
/// 4. IsNotNull(or(x, y)) -> or(IsNotNull(x), y)
///
/// Latter optimizations make sense because calculating IsNull(x)/IsNotNull(x)
/// is at most expensive as calculating x, but usually much cheaper. Although
/// in two cases you exchange 4 computations to 4 computations, the latter
/// four are cheaper.
pub(crate) fn reduce_null<T: ReduceNode>(is_null: bool, node: &T) -> VortexResult<Option<T>> {
    let child = node.child(0);
    if !child.node_dtype()?.is_nullable() {
        return Ok(Some(node.new_constant((!is_null).into())));
    }

    // Irreducible's validity is IsNotNull(self), if we rewrite it here, we'll
    // get into infinite recursion
    if let ReduceNodeValidity::Reduced(validity) = child.validity()? {
        return Ok(Some(if is_null {
            validity.new_node(Not.bind(EmptyOptions), std::slice::from_ref(&validity))?
        } else {
            validity
        }));
    }

    let Some(child_fn) = child.scalar_fn() else {
        return Ok(None);
    };
    let Some(operator) = child_fn.as_opt::<Binary>() else {
        return Ok(None);
    };
    let is_and = match operator {
        Operator::And => true,
        Operator::Or => false,
        _ => return Ok(None),
    };

    let left = child.child(0);
    let right = child.child(1);
    let (nullable, non_nullable) = match (
        left.node_dtype()?.is_nullable(),
        right.node_dtype()?.is_nullable(),
    ) {
        (true, false) => (left, right),
        (false, true) => (right, left),
        // (false, false) already rewritten by constant folding
        _ => return Ok(None),
    };

    // is_null(nullable) (rule 1, 2)
    // is_not_null(nullable) (3, 4)
    let left = if is_null {
        nullable.new_node(IsNull.bind(EmptyOptions), std::slice::from_ref(&nullable))?
    } else {
        is_not_null_node(&nullable)?
    };
    // non_nullable (1, 4)
    // not(non_nullable) (2, 3)
    let right = if is_null == is_and {
        non_nullable
    } else {
        non_nullable.new_node(Not.bind(EmptyOptions), std::slice::from_ref(&non_nullable))?
    };
    let combine = if is_null { Operator::And } else { Operator::Or };
    Ok(Some(node.new_node(Binary.bind(combine), &[left, right])?))
}

/// If we get a scalar function child, at this point we have done all symbolic
/// reductions. Execute child one step and return. For every other child,
/// return the child itself.
pub(crate) fn lazy_child_or_execute_step(
    child: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if child.is::<ScalarFn>() {
        child.execute::<ArrayRef>(ctx)
    } else {
        Ok(child)
    }
}

/// Expression that checks for non-null values.
#[derive(Clone)]
pub struct IsNotNull;

impl IsNotNull {
    /// Creates a lazy non-null check over `input`.
    #[expect(clippy::new_ret_no_self, reason = "constructs the lazy result array")]
    pub fn new(input: ArrayRef) -> ScalarFnArray {
        ScalarFnArray::try_new(IsNotNull.bind(EmptyOptions), vec![input])
            .vortex_expect("IsNotNull has one child and an infallible return dtype")
    }
}

impl ScalarFnVTable for IsNotNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.is_not_null");
        *ID
    }

    fn serialize(&self, _instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for IsNotNull expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "is_not_null(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        _data: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = lazy_child_or_execute_step(args.get(0)?, ctx)?;
        match child.validity()? {
            Validity::NonNullable | Validity::AllValid => {
                Ok(ConstantArray::new(true, args.row_count()).into_array())
            }
            Validity::AllInvalid => Ok(ConstantArray::new(false, args.row_count()).into_array()),
            Validity::Array(a) => Ok(a),
        }
    }

    fn reduce<T: ReduceNode>(&self, _options: &Self::Options, node: &T) -> VortexResult<Option<T>> {
        reduce_null(false, node)
    }

    fn is_strict(&self, _instance: &Self::Options) -> bool {
        // Null input produces the non-null boolean value `false`.
        false
    }

    fn is_infallible(&self, _instance: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::StructFields;
    use crate::expr::BoundExpression;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::is_not_null;
    use crate::expr::is_null;
    use crate::expr::lit;
    use crate::expr::not;
    use crate::expr::or;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::internal::row_count::RowCount;
    use crate::stats::StatsSession;
    use crate::stats::all_null;
    use crate::stats::null_count;

    static STATS_SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            is_not_null(root()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    fn bool_dtype() -> DType {
        let names = ["a", "b"].into();
        let fields = vec![
            DType::Bool(Nullability::Nullable),
            DType::Bool(Nullability::NonNullable),
        ];
        DType::Struct(StructFields::new(names, fields), Nullability::NonNullable)
    }

    fn optimized(expr: Expression, scope: &DType) -> VortexResult<BoundExpression> {
        expr.bind(scope)?.optimize()
    }

    #[test]
    fn reduce_to_constant() -> VortexResult<()> {
        let dtype = bool_dtype();
        assert_eq!(
            optimized(is_not_null(col("b")), &dtype)?,
            lit(true).bind(&dtype)?
        );
        assert_eq!(
            optimized(is_null(col("b")), &dtype)?,
            lit(false).bind(&dtype)?
        );
        Ok(())
    }

    #[test]
    fn reduce_kleene() -> VortexResult<()> {
        let dtype = bool_dtype();
        assert_eq!(
            optimized(is_not_null(and(col("a"), col("b"))), &dtype)?,
            or(is_not_null(col("a")), not(col("b"))).bind(&dtype)?
        );
        assert_eq!(
            optimized(is_not_null(or(col("a"), col("b"))), &dtype)?,
            or(is_not_null(col("a")), col("b")).bind(&dtype)?
        );
        assert_eq!(
            optimized(is_null(and(col("a"), col("b"))), &dtype)?,
            and(is_null(col("a")), col("b")).bind(&dtype)?
        );
        assert_eq!(
            optimized(is_null(or(col("a"), col("b"))), &dtype)?,
            and(is_null(col("a")), not(col("b"))).bind(&dtype)?
        );
        Ok(())
    }

    #[test]
    fn replace_children() {
        let expr = is_not_null(root());
        expr.with_children([root()])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate_mask() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array();
        let expected = [true, false, true, false, true];

        let result = test_array.clone().apply(&is_not_null(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_true() {
        let test_array = buffer![1, 2, 3, 4, 5].into_array();

        let result = test_array.clone().apply(&is_not_null(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        for i in 0..result.len() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                Scalar::bool(true, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_false() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![None::<i32>, None, None, None, None])
                .into_array();

        let result = test_array.clone().apply(&is_not_null(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        for i in 0..result.len() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                Scalar::bool(false, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_struct() {
        let test_array = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array(),
        )])
        .unwrap()
        .into_array();
        let expected = [true, false, true, false, true];

        let result = test_array
            .clone()
            .apply(&is_not_null(get_item("a", root())))
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn test_display() {
        let expr = is_not_null(get_item("name", root()));
        assert_eq!(expr.to_string(), "is_not_null($.name)");

        let expr2 = is_not_null(root());
        assert_eq!(expr2.to_string(), "is_not_null($)");
    }

    #[test]
    fn test_is_not_null_is_not_strict() {
        assert!(
            !is_not_null(col("a"))
                .as_scalar()
                .is_some_and(|f| f.signature().is_strict())
        );
    }

    #[test]
    fn test_is_not_null_falsification() -> VortexResult<()> {
        let expr = is_not_null(col("a"));
        let dtype = test_harness::struct_dtype();

        assert_eq!(
            expr.bind(&dtype)?.falsify(&STATS_SESSION)?,
            Some(
                or(
                    eq(null_count(col("a")), RowCount.new_expr(EmptyOptions, []),),
                    all_null(col("a")),
                )
                .bind(&dtype)?
            )
        );
        Ok(())
    }
}
