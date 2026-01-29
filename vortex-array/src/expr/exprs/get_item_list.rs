// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::ops::Not;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::IntoArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::StructArray;
use crate::compute::mask;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::vtable::ValidityHelper;

/// UNSTABLE: project a struct field from each element of a list.
///
/// Semantics:
/// `get_item_list(field, list) == map(lambda x: get_item(field, x), list)`.
///
/// This is a temporary internal expression used to support nested projections like `items.a` on
/// `list<struct{...}>` and `fixed_size_list<struct{...}>` without a general `map` expression.
///
/// Do not serialize or persist this expression. It is not a stable part of the expression wire
/// format and may be removed or replaced by a proper `map`.
pub struct GetItemList;

impl VTable for GetItemList {
    type Options = FieldName;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.get_item_list")
    }

    fn serialize(&self, _field_name: &FieldName) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("UNSTABLE expression {} must not be serialized", self.id())
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
        _ = metadata;
        vortex_bail!("UNSTABLE expression {} must not be deserialized", self.id())
    }

    fn arity(&self, _field_name: &FieldName) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _field_name: &FieldName, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            _ => unreachable!(
                "Invalid child index {} for GetItemList expression",
                child_idx
            ),
        }
    }

    fn fmt_sql(
        &self,
        field_name: &FieldName,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        expr.child(0).fmt_sql(f)?;
        write!(f, ".{}", field_name)
    }

    fn return_dtype(&self, field_name: &FieldName, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let list_dtype = &arg_dtypes[0];

        let (element_dtype, list_nullability, list_size) = match list_dtype {
            DType::List(element_dtype, list_nullability) => {
                (element_dtype.as_ref(), *list_nullability, None)
            }
            DType::FixedSizeList(element_dtype, list_size, list_nullability) => {
                (element_dtype.as_ref(), *list_nullability, Some(*list_size))
            }
            _ => {
                return Err(vortex_err!(
                    "Expected list dtype for child of GetItemList expression, got {}",
                    list_dtype
                ));
            }
        };

        let struct_fields = element_dtype.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!(
                "Expected list element struct dtype for GetItemList, got {}",
                element_dtype
            )
        })?;

        let field_dtype = struct_fields.field(field_name).ok_or_else(|| {
            vortex_err!(
                "Couldn't find the {} field in the list element struct dtype",
                field_name
            )
        })?;

        let projected = field_dtype.union_nullability(element_dtype.nullability());

        Ok(match list_size {
            Some(list_size) => {
                DType::FixedSizeList(Arc::new(projected), list_size, list_nullability)
            }
            None => DType::List(Arc::new(projected), list_nullability),
        })
    }

    fn execute(
        &self,
        field_name: &FieldName,
        mut args: ExecutionArgs,
    ) -> VortexResult<ExecutionResult> {
        let input = args
            .inputs
            .pop()
            .vortex_expect("missing list for GetItemList expression");

        match input.dtype() {
            DType::List(..) => {
                let list = input.execute::<ListViewArray>(args.ctx)?;
                let struct_elems = list.elements().clone().execute::<StructArray>(args.ctx)?;

                let field = struct_elems.unmasked_field_by_name(field_name)?.clone();
                let field = match struct_elems.dtype().nullability() {
                    Nullability::NonNullable => field,
                    Nullability::Nullable => mask(&field, &struct_elems.validity_mask()?.not())?,
                };

                ListViewArray::try_new(
                    field,
                    list.offsets().clone(),
                    list.sizes().clone(),
                    list.validity().clone(),
                )?
                .into_array()
                .execute(args.ctx)
            }
            DType::FixedSizeList(..) => {
                let list = input.execute::<FixedSizeListArray>(args.ctx)?;
                let struct_elems = list.elements().clone().execute::<StructArray>(args.ctx)?;

                let field = struct_elems.unmasked_field_by_name(field_name)?.clone();
                let field = match struct_elems.dtype().nullability() {
                    Nullability::NonNullable => field,
                    Nullability::Nullable => mask(&field, &struct_elems.validity_mask()?.not())?,
                };

                FixedSizeListArray::try_new(
                    field,
                    list.list_size(),
                    list.validity().clone(),
                    list.len(),
                )?
                .into_array()
                .execute(args.ctx)
            }
            _ => Err(vortex_err!(
                "Expected list scope for GetItemList execution, got {}",
                input.dtype()
            )),
        }
    }

    fn is_null_sensitive(&self, _field_name: &FieldName) -> bool {
        true
    }

    fn is_fallible(&self, _field_name: &FieldName) -> bool {
        false
    }
}

/// Creates an expression that projects a struct field from each element of a list.
#[doc(hidden)]
pub fn get_item_list(field: impl Into<FieldName>, list: Expression) -> Expression {
    GetItemList.new_expr(field.into(), vec![list])
}
