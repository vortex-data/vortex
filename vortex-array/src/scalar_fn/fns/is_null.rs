// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::eq;
use crate::expr::lit;
use crate::expr::stats::Stat;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::validity::Validity;

/// Expression that checks for null values.
#[derive(Clone)]
pub struct IsNull;

impl ScalarFnVTable for IsNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.is_null")
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
            _ => unreachable!("Invalid child index {} for IsNull expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "is_null(")?;
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
        if let Some(scalar) = child.as_constant() {
            return Ok(ConstantArray::new(scalar.is_null(), args.row_count()).into_array());
        }

        match child.validity()? {
            Validity::NonNullable | Validity::AllValid => {
                Ok(ConstantArray::new(false, args.row_count()).into_array())
            }
            Validity::AllInvalid => Ok(ConstantArray::new(true, args.row_count()).into_array()),
            Validity::Array(a) => a.not(),
        }
    }

    fn stat_falsification(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        let null_count_expr = expr.child(0).stat_expression(Stat::NullCount, catalog)?;
        Some(eq(null_count_expr, lit(0u64)))
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _instance: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;
    use vortex_utils::aliases::hash_map::HashMap;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Field;
    use crate::dtype::FieldPath;
    use crate::dtype::FieldPathSet;
    use crate::dtype::Nullability;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::is_null;
    use crate::expr::lit;
    use crate::expr::pruning::checked_pruning_expr;
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            is_null(root()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = is_null(root());
        expr.with_children([root()])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate_mask() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array();
        let expected = [false, true, false, true, false];

        let result = test_array.clone().apply(&is_null(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_false() {
        let test_array = buffer![1, 2, 3, 4, 5].into_array();

        let result = test_array.clone().apply(&is_null(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        // All values should be false (non-nullable input)
        for i in 0..result.len() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::bool(false, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_true() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![None::<i32>, None, None, None, None])
                .into_array();

        let result = test_array.clone().apply(&is_null(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        // All values should be true (all nulls)
        for i in 0..result.len() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::bool(true, Nullability::NonNullable)
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
        let expected = [false, true, false, true, false];

        let result = test_array
            .clone()
            .apply(&is_null(get_item("a", root())))
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn test_display() {
        let expr = is_null(get_item("name", root()));
        assert_eq!(expr.to_string(), "is_null($.name)");

        let expr2 = is_null(root());
        assert_eq!(expr2.to_string(), "is_null($)");
    }

    #[test]
    fn test_is_null_falsification() {
        let expr = is_null(col("a"));

        let (pruning_expr, st) = checked_pruning_expr(
            &expr,
            &FieldPathSet::from_iter([FieldPath::from_iter([
                Field::Name("a".into()),
                Field::Name("null_count".into()),
            ])]),
        )
        .unwrap();

        assert_eq!(&pruning_expr, &eq(col("a_null_count"), lit(0u64)));
        assert_eq!(
            st.map(),
            &HashMap::from_iter([(FieldPath::from_name("a"), HashSet::from([Stat::NullCount]))])
        );
    }

    #[test]
    fn test_is_null_sensitive() {
        // is_null itself is null-sensitive
        assert!(is_null(col("a")).signature().is_null_sensitive());
    }
}
