// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::StructArray;
use crate::builtins::ArrayBuiltins;
use crate::builtins::ExprBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldPath;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::lit;
use crate::expr::stats::Stat;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::pack::Pack;

#[derive(Clone)]
pub struct GetItem;

impl ScalarFnVTable for GetItem {
    type Options = FieldName;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.get_item")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::GetItemOpts {
                path: instance.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::GetItemOpts::decode(_metadata)?;
        Ok(FieldName::from(opts.path))
    }

    fn arity(&self, _field_name: &FieldName) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for GetItem expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        field_name: &FieldName,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        expr.children()[0].fmt_sql(f)?;
        write!(f, ".{}", field_name)
    }

    fn return_dtype(&self, field_name: &FieldName, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let struct_dtype = &arg_dtypes[0];
        let field_dtype = struct_dtype
            .as_struct_fields_opt()
            .and_then(|st| st.field(field_name))
            .ok_or_else(|| {
                vortex_err!("Couldn't find the {} field in the input scope", field_name)
            })?;

        // Match here to avoid cloning the dtype if nullability doesn't need to change
        if matches!(
            (struct_dtype.nullability(), field_dtype.nullability()),
            (Nullability::Nullable, Nullability::NonNullable)
        ) {
            return Ok(field_dtype.with_nullability(Nullability::Nullable));
        }

        Ok(field_dtype)
    }

    fn execute(
        &self,
        field_name: &FieldName,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?.execute::<StructArray>(ctx)?;
        let field = input.unmasked_field_by_name(field_name).cloned()?;

        match input.dtype().nullability() {
            Nullability::NonNullable => Ok(field),
            Nullability::Nullable => field.mask(input.validity().to_array(input.len())),
        }
    }

    fn reduce(
        &self,
        field_name: &FieldName,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        let child = node.child(0);
        if let Some(child_fn) = child.scalar_fn()
            && let Some(pack) = child_fn.as_opt::<Pack>()
            && let Some(idx) = pack.names.find(field_name)
        {
            let mut field = child.child(idx);

            // Possibly mask the field if the pack is nullable
            if pack.nullability.is_nullable() {
                field = ctx.new_node(
                    Mask.bind(EmptyOptions),
                    &[field, ctx.new_node(Literal.bind(true.into()), &[])?],
                )?;
            }

            return Ok(Some(field));
        }

        Ok(None)
    }

    fn simplify_untyped(
        &self,
        field_name: &FieldName,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let child = expr.child(0);

        // If the child is a Pack expression, we can directly return the corresponding child.
        if let Some(pack) = child.as_opt::<Pack>() {
            let idx = pack
                .names
                .iter()
                .position(|name| name == field_name)
                .ok_or_else(|| {
                    vortex_err!(
                        "Cannot find field {} in pack fields {:?}",
                        field_name,
                        pack.names
                    )
                })?;

            let mut field = child.child(idx).clone();

            // It's useful to simplify this node without type info, but we need to make sure
            // the nullability is correct. We cannot cast since we don't have the dtype info here,
            // so instead we insert a Mask expression that we know converts a child's dtype to
            // nullable.
            if pack.nullability.is_nullable() {
                // Mask with an all-true array to ensure the field DType is nullable.
                field = field.mask(lit(true))?;
            }

            return Ok(Some(field));
        }

        Ok(None)
    }

    fn stat_expression(
        &self,
        field_name: &FieldName,
        _expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // TODO(ngates): I think we can do better here and support stats over nested fields.
        //  It would be nice if delegating to our child would return a struct of statistics
        //  matching the nested DType such that we can write:
        //    `get_item(expr.child(0).stat_expression(...), expr.data().field_name())`

        // TODO(ngates): this is a bug whereby we may return stats for a nested field of the same
        //  name as a field in the root struct. This should be resolved with upcoming change to
        //  falsify expressions, but for now I'm preserving the existing buggy behavior.
        catalog.stats_ref(&FieldPath::from_name(field_name.clone()), stat)
    }

    // This will apply struct nullability field. We could add a dtype??
    fn is_null_sensitive(&self, _field_name: &FieldName) -> bool {
        true
    }

    fn is_fallible(&self, _field_name: &FieldName) -> bool {
        // If this type-checks its infallible.
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::checked_add;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::scalar_fn::fns::get_item::StructArray;
    use crate::validity::Validity;

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
    }

    #[test]
    fn get_item_by_name() {
        let st = test_array();
        let get_item = get_item("a", root());
        let item = st.into_array().apply(&get_item).unwrap();
        assert_eq!(item.dtype(), &DType::from(PType::I32))
    }

    #[test]
    fn get_item_by_name_none() {
        let st = test_array();
        let get_item = get_item("c", root());
        assert!(st.into_array().apply(&get_item).is_err());
    }

    #[test]
    #[ignore = "apply() has a bug with null propagation from struct validity to non-nullable child fields"]
    fn get_nullable_field() {
        let st = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![buffer![1i32].into_array()],
            1,
            Validity::AllInvalid,
        )
        .unwrap()
        .into_array();

        let get_item_expr = get_item("a", root());
        let item = st.apply(&get_item_expr).unwrap();
        // The dtype should be nullable since it inherits struct validity
        assert_eq!(
            item.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_pack_get_item_rule() {
        // Create: pack(a: lit(1), b: lit(2)).get_item("b")
        let pack_expr = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let get_item_expr = get_item("b", pack_expr);

        let result = get_item_expr
            .optimize_recursive(&DType::Struct(StructFields::empty(), NonNullable))
            .unwrap();

        assert_eq!(result, lit(2));
    }

    #[test]
    fn test_multi_level_pack_get_item_simplify() {
        let inner_pack = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let get_a = get_item("a", inner_pack);

        let outer_pack = pack([("x", get_a), ("y", lit(3)), ("z", lit(4))], NonNullable);
        let get_z = get_item("z", outer_pack);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        let result = get_z.optimize_recursive(&dtype).unwrap();
        assert_eq!(result, lit(4));
    }

    #[test]
    fn test_deeply_nested_pack_get_item() {
        let innermost = pack([("a", lit(42))], NonNullable);
        let get_a = get_item("a", innermost);

        let level2 = pack([("b", get_a)], NonNullable);
        let get_b = get_item("b", level2);

        let level3 = pack([("c", get_b)], NonNullable);
        let get_c = get_item("c", level3);

        let outermost = pack([("final", get_c)], NonNullable);
        let get_final = get_item("final", outermost);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        let result = get_final.optimize_recursive(&dtype).unwrap();
        assert_eq!(result, lit(42));
    }

    #[test]
    fn test_partial_pack_get_item_simplify() {
        let inner_pack = pack([("x", lit(1)), ("y", lit(2))], NonNullable);
        let get_x = get_item("x", inner_pack);
        let add_expr = checked_add(get_x, lit(10));

        let outer_pack = pack([("result", add_expr)], NonNullable);
        let get_result = get_item("result", outer_pack);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        let result = get_result.optimize_recursive(&dtype).unwrap();
        let expected = checked_add(lit(1), lit(10));
        assert_eq!(&result, &expected);
    }

    #[test]
    fn get_item_filter_list_field() {
        use vortex_mask::Mask;

        use crate::arrays::BoolArray;
        use crate::arrays::FilterArray;
        use crate::arrays::ListArray;

        let list = ListArray::try_new(
            buffer![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.].into_array(),
            buffer![2u64, 4, 6, 8, 10, 12].into_array(),
            Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array()),
        )
        .unwrap();

        let filtered = FilterArray::try_new(
            list.into_array(),
            Mask::from_iter([true, true, false, false, false]),
        )
        .unwrap();

        let st = StructArray::try_new(
            FieldNames::from(["data"]),
            vec![filtered.into_array()],
            2,
            Validity::AllValid,
        )
        .unwrap();

        st.into_array().apply(&get_item("data", root())).unwrap();
    }
}
