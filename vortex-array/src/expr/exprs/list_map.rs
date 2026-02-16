// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use prost::Message;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ListViewArray;
use crate::canonical::ToCanonical;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::exprs::root::Root;
use crate::expr::proto::ExprSerializeProtoExt;

/// The transform expression, stored as options rather than a child.
///
/// This is the key design choice: the transform operates in a *different scope* (the list's
/// element type) than the list child (the outer scope). Storing it in options rather than as
/// a child means tree walkers, field reference collectors, and simplifiers correctly skip it,
/// since its `root()` refers to list elements, not the outer scope.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ListMapOptions {
    pub transform: Expression,
}

impl fmt::Display for ListMapOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.transform.fmt_sql(f)
    }
}

/// Expression vtable for mapping a transform over the elements of a list.
///
/// `list_map(list_expr, element_transform)` applies `element_transform` to the flat
/// elements array of a list, rebinding `root()` to refer to each element. This is a
/// columnar operation — the transform runs once on the entire flat elements array, then
/// the list is reconstructed with the same offsets/sizes/validity.
///
/// The transform is stored in [`ListMapOptions`] rather than as a child expression because
/// it operates in a different scope (the list element type). This avoids special-casing in
/// the expression dispatch paths.
pub struct ListMap;

impl VTable for ListMap {
    type Options = ListMapOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.list.map")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let proto = options.transform.serialize_proto()?;
        Ok(Some(proto.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let proto = pb::Expr::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode ListMap transform: {}", e))?;
        let transform = Expression::from_proto(&proto, session)?;
        Ok(ListMapOptions { transform })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            _ => unreachable!("Invalid child index {} for ListMap expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ").map(|$| ")?;
        options.transform.fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let list_dtype = &arg_dtypes[0];
        let element_dtype = list_dtype.as_list_element_opt().ok_or_else(|| {
            vortex_err!("ListMap child must produce a list type, got {}", list_dtype)
        })?;
        let transformed_element_dtype = options.transform.return_dtype(element_dtype)?;
        Ok(DType::List(
            Arc::new(transformed_element_dtype),
            list_dtype.nullability(),
        ))
    }

    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let list_array = args
            .inputs
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("ListMap expected 1 input"))?;
        let list_view = list_array.to_listview();
        let parts = list_view.into_parts();
        let transformed_elements = parts.elements.apply(&options.transform)?;
        Ok(ListViewArray::try_new(
            transformed_elements,
            parts.offsets,
            parts.sizes,
            parts.validity,
        )?
        .into_array())
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn simplify_untyped(
        &self,
        options: &Self::Options,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // list_map(x, root()) => x  (identity map is a no-op)
        if options.transform.is::<Root>() {
            return Ok(Some(expr.child(0).clone()));
        }
        Ok(None)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Creates an expression that applies a transform to each element of a list.
///
/// The `element_transform` expression is evaluated with `root()` bound to the list's
/// flat elements array. This is a columnar operation — the transform runs once on
/// the entire elements array, not per-element.
///
/// ```rust
/// # use vortex_array::expr::{list_map, root};
/// // Identity map: no-op, simplifies to just `root()`
/// let expr = list_map(root(), root());
/// ```
pub fn list_map(list: Expression, transform: Expression) -> Expression {
    ListMap.new_expr(ListMapOptions { transform }, [list])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::StructFields;
    use vortex_error::VortexResult;

    use super::ListMap;
    use super::ListMapOptions;
    use super::list_map;
    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::expr::VTable;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::pack::pack;
    use crate::expr::exprs::root::root;
    use crate::validity::Validity;

    #[test]
    fn test_identity_simplifies() {
        let expr = list_map(col("list_col"), root());
        let options = ListMapOptions { transform: root() };
        let simplified = ListMap.simplify_untyped(&options, &expr).unwrap();
        assert_eq!(simplified, Some(col("list_col")));
    }

    #[test]
    fn test_return_dtype() -> VortexResult<()> {
        let inner_struct = DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(PType::I64, Nullability::Nullable)],
            ),
            Nullability::NonNullable,
        );
        let scope = DType::Struct(
            StructFields::new(
                ["my_list"].into(),
                vec![DType::List(Arc::new(inner_struct), Nullability::Nullable)],
            ),
            Nullability::NonNullable,
        );

        let transform = pack(
            [
                ("a", get_item("a", root())),
                (
                    "b",
                    lit(vortex_scalar::Scalar::null(DType::Primitive(
                        PType::I64,
                        Nullability::Nullable,
                    ))),
                ),
            ],
            Nullability::NonNullable,
        );

        let expr = list_map(get_item("my_list", root()), transform);
        let result_dtype = expr.return_dtype(&scope)?;

        let expected_element = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Primitive(PType::I64, Nullability::Nullable),
                    DType::Primitive(PType::I64, Nullability::Nullable),
                ],
            ),
            Nullability::NonNullable,
        );
        assert_eq!(
            result_dtype,
            DType::List(Arc::new(expected_element), Nullability::Nullable)
        );

        Ok(())
    }

    #[test]
    fn test_execute_transforms_elements() -> VortexResult<()> {
        let elements = PrimitiveArray::from_iter(vec![1i64, 2, 3, 4, 5]).into_array();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(vec![0i64, 3, 5]).into_array(),
            Validity::AllValid,
        )?
        .into_array();

        let expr = list_map(root(), root());
        let result = list.apply(&expr)?;

        assert!(matches!(result.dtype(), DType::List(..)));
        assert_eq!(result.len(), 2);

        let first_list = result.scalar_at(0)?;
        let first_list_val = first_list.as_list();
        assert_eq!(first_list_val.elements().unwrap().len(), 3);

        let second_list = result.scalar_at(1)?;
        let second_list_val = second_list.as_list();
        assert_eq!(second_list_val.elements().unwrap().len(), 2);

        Ok(())
    }

    #[test]
    #[ignore = "hits pre-existing optimizer StructGetItemRule dtype mismatch on canonical list elements"]
    fn test_execute_struct_transform() -> VortexResult<()> {
        let elements = StructArray::try_new(
            ["a"].into(),
            vec![PrimitiveArray::from_iter(vec![1i64, 2, 3]).into_array()],
            3,
            Validity::AllValid,
        )?
        .into_array();

        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(vec![0i64, 2, 3]).into_array(),
            Validity::AllValid,
        )?
        .into_array();

        let expr = list_map(root(), get_item("a", root()));
        let result = list.apply(&expr)?;

        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
                Nullability::NonNullable,
            )
        );
        assert_eq!(result.len(), 2);

        Ok(())
    }

    #[test]
    fn test_nullability_preserved() -> VortexResult<()> {
        let elements = PrimitiveArray::from_iter(vec![1i64, 2]).into_array();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(vec![0i64, 2, 2]).into_array(),
            Validity::from_iter([true, false]),
        )?
        .into_array();

        let expr = list_map(root(), root());
        let result = list.apply(&expr)?;

        assert!(result.is_valid(0)?);
        assert!(!result.is_valid(1)?);
        assert_eq!(result.len(), 2);

        Ok(())
    }

    #[test]
    fn test_nested_list_map() -> VortexResult<()> {
        let inner_elements = PrimitiveArray::from_iter(vec![1i64, 2, 3, 4]).into_array();
        let inner_lists = ListArray::try_new(
            inner_elements,
            PrimitiveArray::from_iter(vec![0i64, 2, 3, 4]).into_array(),
            Validity::AllValid,
        )?
        .into_array();

        let outer_list = ListArray::try_new(
            inner_lists,
            PrimitiveArray::from_iter(vec![0i64, 2, 3]).into_array(),
            Validity::AllValid,
        )?
        .into_array();

        let inner_transform = list_map(root(), root());
        let expr = list_map(root(), inner_transform);

        let scope = outer_list.dtype().clone();
        let result_dtype = expr.return_dtype(&scope)?;
        assert_eq!(result_dtype, scope);

        let result = outer_list.apply(&expr)?;
        assert_eq!(result.len(), 2);

        let first = result.scalar_at(0)?;
        let first_list = first.as_list();
        assert_eq!(first_list.elements().unwrap().len(), 2);

        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = list_map(get_item("tags", root()), root());
        assert_eq!(expr.to_string(), "($.tags).map(|$| $)");

        let expr2 = list_map(root(), get_item("name", root()));
        assert_eq!(expr2.to_string(), "($).map(|$| $.name)");
    }
}
