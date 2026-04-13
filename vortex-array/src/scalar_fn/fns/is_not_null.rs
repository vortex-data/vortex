// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::and;
use crate::expr::eq;
use crate::expr::gt;
use crate::expr::lit;
use crate::expr::stats::Stat;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::validity::Validity;

/// Expression that checks for non-null values.
#[derive(Clone)]
pub struct IsNotNull;

impl ScalarFnVTable for IsNotNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.is_not_null")
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
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "is_not_null(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        _data: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?;
        match child.validity()? {
            Validity::NonNullable | Validity::AllValid => {
                Ok(ConstantArray::new(true, args.row_count()).into_array())
            }
            Validity::AllInvalid => Ok(ConstantArray::new(false, args.row_count()).into_array()),
            Validity::Array(a) => Ok(a),
        }
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _instance: &Self::Options) -> bool {
        false
    }

    fn stat_falsification(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // is_not_null is falsified when ALL values are null, i.e. null_count == len.
        // Since there is no len stat in the zone map, we approximate using IsConstant:
        // if the zone is constant and has any nulls, then all values must be null.
        //
        // TODO(#7187): Add a len stat to enable the more general falsification:
        //   null_count == len => is_not_null is all false.
        let null_count_expr = expr.child(0).stat_expression(Stat::NullCount, catalog)?;
        let is_constant_expr = expr.child(0).stat_expression(Stat::IsConstant, catalog)?;
        // If the zone is constant (is_constant == true) and has nulls (null_count > 0),
        // then all values must be null, so is_not_null is all false.
        Some(and(
            eq(is_constant_expr, lit(true)),
            gt(null_count_expr, lit(0u64)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::get_item;
    use crate::expr::is_not_null;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            is_not_null(root()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
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
                result.scalar_at(i).unwrap(),
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
                result.scalar_at(i).unwrap(),
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
                result.scalar_at(i).unwrap(),
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
                result.scalar_at(i).unwrap(),
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
    fn test_is_not_null_sensitive() {
        use crate::expr::col;
        assert!(is_not_null(col("a")).signature().is_null_sensitive());
    }

    #[test]
    fn test_is_not_null_falsification() {
        use vortex_utils::aliases::hash_map::HashMap;
        use vortex_utils::aliases::hash_set::HashSet;

        use crate::dtype::Field;
        use crate::dtype::FieldPath;
        use crate::dtype::FieldPathSet;
        use crate::expr::and;
        use crate::expr::col;
        use crate::expr::eq;
        use crate::expr::gt;
        use crate::expr::lit;
        use crate::expr::pruning::checked_pruning_expr;
        use crate::expr::stats::Stat;

        let expr = is_not_null(col("a"));

        let (pruning_expr, st) = checked_pruning_expr(
            &expr,
            &FieldPathSet::from_iter([
                FieldPath::from_iter([Field::Name("a".into()), Field::Name("null_count".into())]),
                FieldPath::from_iter([Field::Name("a".into()), Field::Name("is_constant".into())]),
            ]),
        )
        .unwrap();

        assert_eq!(
            &pruning_expr,
            &and(
                eq(col("a_is_constant"), lit(true)),
                gt(col("a_null_count"), lit(0u64)),
            )
        );
        assert_eq!(
            st.map(),
            &HashMap::from_iter([(
                FieldPath::from_name("a"),
                HashSet::from([Stat::NullCount, Stat::IsConstant])
            )])
        );
    }
}
