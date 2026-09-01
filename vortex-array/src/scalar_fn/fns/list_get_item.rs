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
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::list_length::AnyList;

/// Element-wise field projection through a list: `List<Struct{..., f, ...}>` -> `List<f>`.
///
/// The list shape (offsets/sizes and validity) is carried over unchanged; only the elements
/// child is narrowed to the requested struct field. Struct-level element nullability is folded
/// into the projected field, mirroring [`GetItem`].
#[derive(Clone)]
pub struct ListGetItem;

impl ScalarFnVTable for ListGetItem {
    type Options = FieldName;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.get_item");
        *ID
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
        let (element_dtype, rebuild): (&DType, &dyn Fn(DType) -> DType) = match &arg_dtypes[0] {
            DType::List(element, nullability) => (element.as_ref(), &move |f| {
                DType::List(f.into(), *nullability)
            }),
            DType::FixedSizeList(element, size, nullability) => (element.as_ref(), &move |f| {
                DType::FixedSizeList(f.into(), *size, *nullability)
            }),
            other => vortex_bail!("list_get_item() requires a list input, got {other}"),
        };

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
        let field_dtype = if matches!(
            (element_dtype.nullability(), field_dtype.nullability()),
            (Nullability::Nullable, Nullability::NonNullable)
        ) {
            field_dtype.with_nullability(Nullability::Nullable)
        } else {
            field_dtype
        };

        Ok(rebuild(field_dtype))
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
            let projected = GetItem::try_new(l.elements().clone(), field_name.clone())?;
            Ok(ListArray::try_new(
                projected.into_array(),
                l.offsets().clone(),
                l.list_validity(),
            )?
            .into_array())
        } else if let Some(lv) = any_list.as_opt::<ListView>() {
            let projected = GetItem::try_new(lv.elements().clone(), field_name.clone())?;
            Ok(ListViewArray::try_new(
                projected.into_array(),
                lv.offsets().clone(),
                lv.sizes().clone(),
                lv.listview_validity(),
            )?
            .into_array())
        } else if let Some(fsl) = any_list.as_opt::<FixedSizeList>() {
            let projected = GetItem::try_new(fsl.elements().clone(), field_name.clone())?;
            let len = fsl.as_ref().len();
            Ok(FixedSizeListArray::new(
                projected.into_array(),
                fsl.list_size(),
                fsl.validity()?,
                len,
            )
            .into_array())
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
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::list_get_item;
    use crate::expr::root;
    use crate::validity::Validity;

    /// 4 struct elements {a: i32, b: i64} in 3 lists of lengths [2, 1, 1].
    fn create_list_of_structs(validity: Validity) -> ArrayRef {
        let elements = StructArray::from_fields(&[
            ("a", buffer![10i32, 11, 20, 30].into_array()),
            ("b", buffer![-1i64, -2, -3, -4].into_array()),
        ])
        .unwrap()
        .into_array();
        ListArray::try_new(elements, buffer![0u32, 2, 3, 4].into_array(), validity)
            .unwrap()
            .into_array()
    }

    #[test]
    fn projects_field_and_keeps_shape() -> VortexResult<()> {
        let list = create_list_of_structs(Validity::NonNullable);
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
        let list = create_list_of_structs(validity.clone());
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
        let elements = StructArray::from_fields(&[
            ("a", buffer![10i32, 11, 20, 30].into_array()),
            ("b", buffer![-1i64, -2, -3, -4].into_array()),
        ])?
        .into_array();
        let lv = ListViewArray::try_new(
            elements,
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
    fn unknown_field_fails_to_bind() {
        let list = create_list_of_structs(Validity::NonNullable);
        assert!(list.apply(&list_get_item("missing", root())).is_err());
    }

    #[test]
    fn test_display() {
        let expr = list_get_item("a", root());
        assert_eq!(expr.to_string(), "$[].a");
    }
}
