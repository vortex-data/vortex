// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::display::ExprDisplay;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::list_length::AnyList;

/// Element-wise field projection through a list: `List<Struct{..., f, ...}>` -> `List<f>`.
///
/// The list shape (offsets/sizes and validity) is carried over unchanged; only the elements
/// child is narrowed to the requested struct field. Struct-level element nullability is folded
/// into the projected field, mirroring [`GetItem`]. Nested lists project recursively:
/// `List<List<Struct{..., f, ...}>>` -> `List<List<f>>`, so a chain of `list_get_item`
/// descends a struct/list tree the way a Parquet leaf path does.
#[derive(Clone)]
pub struct ListGetItem;

impl ListGetItem {
    /// Creates a lazy element-wise projection of `field_name` through the lists of `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not list-typed or its (transitively) innermost element
    /// is not a struct containing `field_name`.
    pub fn try_new(
        input: ArrayRef,
        field_name: impl Into<FieldName>,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(ListGetItem.bind(field_name.into()), vec![input])
    }
}

/// Project `field_name` out of a list's element dtype, descending nested lists.
fn project_element_dtype(element_dtype: &DType, field_name: &FieldName) -> VortexResult<DType> {
    match element_dtype {
        DType::List(inner, nullability) => Ok(DType::List(
            project_element_dtype(inner, field_name)?.into(),
            *nullability,
        )),
        DType::FixedSizeList(inner, size, nullability) => Ok(DType::FixedSizeList(
            project_element_dtype(inner, field_name)?.into(),
            *size,
            *nullability,
        )),
        _ => {
            let field_dtype = element_dtype
                .as_struct_fields_opt()
                .and_then(|st| st.field(field_name))
                .ok_or_else(|| {
                    vortex_err!(
                        "list_get_item() couldn't find field {} in list element {}",
                        field_name,
                        element_dtype
                    )
                })?;

            // A nullable struct element makes the projected field nullable, mirroring GetItem.
            if matches!(
                (element_dtype.nullability(), field_dtype.nullability()),
                (Nullability::Nullable, Nullability::NonNullable)
            ) {
                Ok(field_dtype.with_nullability(Nullability::Nullable))
            } else {
                Ok(field_dtype)
            }
        }
    }
}

/// Lazily project `field_name` from a list's elements child: through a `GetItem` for struct
/// elements, or a recursive `ListGetItem` for nested-list elements.
fn project_elements(elements: &ArrayRef, field_name: &FieldName) -> VortexResult<ArrayRef> {
    match elements.dtype() {
        DType::List(..) | DType::FixedSizeList(..) => {
            Ok(ListGetItem::try_new(elements.clone(), field_name.clone())?.into_array())
        }
        _ => Ok(GetItem::try_new(elements.clone(), field_name.clone())?.into_array()),
    }
}

impl ScalarFnVTable for ListGetItem {
    type Options = FieldName;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.get_item");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        // The options are a single field name, identical to `GetItem`'s, so the same proto
        // message is reused; the fn id keeps the two distinct on the wire.
        Ok(Some(
            pb::GetItemOpts {
                path: instance.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::GetItemOpts::decode(metadata)?;
        Ok(FieldName::from(opts.path))
    }

    fn arity(&self, _field_name: &FieldName) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_get_item()"),
        }
    }

    fn fmt_sql(
        &self,
        field_name: &FieldName,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        Display::fmt(expr.display_child(0), f)?;
        write!(f, "[].{}", field_name)
    }

    fn return_dtype(&self, field_name: &FieldName, arg_dtypes: &[DType]) -> VortexResult<DType> {
        match &arg_dtypes[0] {
            DType::List(element, nullability) => Ok(DType::List(
                project_element_dtype(element, field_name)?.into(),
                *nullability,
            )),
            DType::FixedSizeList(element, size, nullability) => Ok(DType::FixedSizeList(
                project_element_dtype(element, field_name)?.into(),
                *size,
                *nullability,
            )),
            other => vortex_bail!("list_get_item() requires a list input, got {other}"),
        }
    }

    fn execute(
        &self,
        field_name: &FieldName,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let any_list = input.execute_until::<AnyList>(ctx)?;

        if let Some(l) = any_list.as_opt::<List>() {
            let projected = project_elements(l.elements(), field_name)?;
            Ok(ListArray::try_new(projected, l.offsets().clone(), l.list_validity())?.into_array())
        } else if let Some(lv) = any_list.as_opt::<ListView>() {
            let projected = project_elements(lv.elements(), field_name)?;
            Ok(ListViewArray::try_new(
                projected,
                lv.offsets().clone(),
                lv.sizes().clone(),
                lv.listview_validity(),
            )?
            .into_array())
        } else if let Some(fsl) = any_list.as_opt::<FixedSizeList>() {
            let projected = project_elements(fsl.elements(), field_name)?;
            let len = fsl.as_ref().len();
            Ok(
                FixedSizeListArray::new(projected, fsl.list_size(), fsl.validity()?, len)
                    .into_array(),
            )
        } else {
            let dtype = any_list.dtype();
            vortex_bail!("list_get_item() requires List, ListView, or FixedSizeList, got {dtype}")
        }
    }

    fn validity(
        &self,
        _field_name: &FieldName,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // Row validity is the list validity, carried over from the input unchanged.
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_strict(&self, _field_name: &FieldName) -> bool {
        true
    }

    fn is_infallible(&self, _field_name: &FieldName) -> bool {
        // If this type-checks, it is infallible.
        true
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_proto::expr as pb;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::list_get_item;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::validity::Validity;

    /// 4 struct elements {a: i32, b: i64} in 3 lists of lengths [2, 1, 1].
    fn create_list_of_structs(validity: Validity) -> VortexResult<ArrayRef> {
        let elements = create_struct_elements()?;
        Ok(
            ListArray::try_new(elements, buffer![0u32, 2, 3, 4].into_array(), validity)?
                .into_array(),
        )
    }

    /// 4 struct elements {a: i32, b: i64}.
    fn create_struct_elements() -> VortexResult<ArrayRef> {
        Ok(StructArray::from_fields(&[
            ("a", buffer![10i32, 11, 20, 30].into_array()),
            ("b", buffer![-1i64, -2, -3, -4].into_array()),
        ])?
        .into_array())
    }

    #[test]
    fn projects_field_and_keeps_shape() -> VortexResult<()> {
        let list = create_list_of_structs(Validity::NonNullable)?;
        let result = list.apply(&list_get_item("a", root()))?;

        assert_eq!(
            result.dtype(),
            &DType::List(
                DType::Primitive(PType::I32, Nullability::NonNullable).into(),
                Nullability::NonNullable
            )
        );

        let expected = ListArray::try_new(
            buffer![10i32, 11, 20, 30].into_array(),
            buffer![0u32, 2, 3, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn carries_list_validity() -> VortexResult<()> {
        let validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let list = create_list_of_structs(validity.clone())?;
        let result = list.apply(&list_get_item("b", root()))?;

        let expected = ListArray::try_new(
            buffer![-1i64, -2, -3, -4].into_array(),
            buffer![0u32, 2, 3, 4].into_array(),
            validity,
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn projects_through_listview() -> VortexResult<()> {
        let lv = ListViewArray::try_new(
            create_struct_elements()?,
            buffer![0u32, 2, 3].into_array(),
            buffer![2u32, 1, 1].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let result = lv.apply(&list_get_item("a", root()))?;

        let expected = ListArray::try_new(
            buffer![10i32, 11, 20, 30].into_array(),
            buffer![0u32, 2, 3, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn projects_through_nested_lists() -> VortexResult<()> {
        // List<List<Struct{a, b}>>: outer offsets [0, 2, 3] over inner lists [0, 2, 3, 4].
        let inner = create_list_of_structs(Validity::NonNullable)?;
        let outer = ListArray::try_new(
            inner,
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let result = outer.apply(&list_get_item("a", root()))?;

        let expected_inner = ListArray::try_new(
            buffer![10i32, 11, 20, 30].into_array(),
            buffer![0u32, 2, 3, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let expected = ListArray::try_new(
            expected_inner,
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        assert_eq!(result.dtype(), expected.dtype());
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn projects_through_fixed_size_list() -> VortexResult<()> {
        // 2 lists of size 2 over the 4 struct elements.
        let fsl = FixedSizeListArray::new(create_struct_elements()?, 2, Validity::NonNullable, 2)
            .into_array();
        let result = fsl.apply(&list_get_item("b", root()))?;

        let expected = FixedSizeListArray::new(
            buffer![-1i64, -2, -3, -4].into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();
        assert_eq!(result.dtype(), expected.dtype());
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn unknown_field_fails_to_bind() -> VortexResult<()> {
        let list = create_list_of_structs(Validity::NonNullable)?;
        assert!(list.apply(&list_get_item("missing", root())).is_err());
        Ok(())
    }

    #[test]
    fn test_proto_round_trip() -> VortexResult<()> {
        let expr = list_get_item("a", list_get_item("inner", root()));
        let buf = expr.serialize_proto()?.encode_to_vec();
        let decoded = pb::Expr::decode(buf.as_slice())?;
        assert_eq!(expr, Expression::from_proto(&decoded, &array_session())?);
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = list_get_item("a", root());
        assert_eq!(expr.to_string(), "$[].a");
    }
}
