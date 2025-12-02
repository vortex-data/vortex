// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod transform;

use std::fmt::Formatter;
use std::ops::Not;

use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldPath;
use vortex_dtype::Nullability;
use vortex_error::vortex_err;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_proto::expr as pb;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorOps;

use crate::compute::mask;
use crate::expr::exprs::root::root;
use crate::expr::stats::Stat;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::Pack;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::ArrayRef;
use crate::ToCanonical;

pub struct GetItem;

impl VTable for GetItem {
    type Options = FieldName;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.get_item")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::GetItemOpts {
                path: instance.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
        let opts = pb::GetItemOpts::decode(metadata)?;
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

    fn evaluate(
        &self,
        field_name: &FieldName,
        expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let input = expr.children()[0].evaluate(scope)?.to_struct();
        let field = input.field_by_name(field_name).cloned()?;

        match input.dtype().nullability() {
            Nullability::NonNullable => Ok(field),
            Nullability::Nullable => mask(&field, &input.validity_mask().not()),
        }
    }

    fn execute(&self, field_name: &FieldName, mut args: ExecutionArgs) -> VortexResult<Datum> {
        let struct_dtype = args.dtypes[0]
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype for child of GetItem expression"))?;
        let field_idx = struct_dtype
            .find(field_name)
            .ok_or_else(|| vortex_err!("Field {} not found in struct dtype", field_name))?;

        match args.datums.pop().vortex_expect("missing input") {
            Datum::Scalar(s) => {
                let mut field = s.as_struct().field(field_idx);
                field.mask_validity(s.is_valid());
                Ok(Datum::Scalar(field))
            }
            Datum::Vector(v) => {
                let mut field = v.as_struct().fields()[field_idx].clone();
                field.mask_validity(v.validity());
                Ok(Datum::Vector(field))
            }
        }
    }

    fn simplify(
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

            return Ok(Some(child.child(idx).clone()));
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

/// Creates an expression that accesses a field from the root array.
///
/// Equivalent to `get_item(field, root())` - extracts a named field from the input array.
///
/// ```rust
/// # use vortex_array::expr::col;
/// let expr = col("name");
/// ```
pub fn col(field: impl Into<FieldName>) -> Expression {
    GetItem.new_expr(field.into(), vec![root()])
}

/// Creates an expression that extracts a named field from a struct expression.
///
/// Accesses the specified field from the result of the child expression.
///
/// ```rust
/// # use vortex_array::expr::{get_item, root};
/// let expr = get_item("user_id", root());
/// ```
pub fn get_item(field: impl Into<FieldName>, child: Expression) -> Expression {
    GetItem.new_expr(field.into(), vec![child])
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::FieldNames;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType::I32;
    use vortex_scalar::Scalar;

    use super::get_item;
    use crate::arrays::StructArray;
    use crate::expr::exprs::root::root;
    use crate::validity::Validity;
    use crate::Array;
    use crate::IntoArray;

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
        let item = get_item.evaluate(&st.to_array()).unwrap();
        assert_eq!(item.dtype(), &DType::from(I32))
    }

    #[test]
    fn get_item_by_name_none() {
        let st = test_array();
        let get_item = get_item("c", root());
        assert!(get_item.evaluate(&st.to_array()).is_err());
    }

    #[test]
    fn get_nullable_field() {
        let st = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![buffer![1i32].into_array()],
            1,
            Validity::AllInvalid,
        )
        .unwrap()
        .to_array();

        let get_item = get_item("a", root());
        let item = get_item.evaluate(&st).unwrap();
        assert_eq!(
            item.scalar_at(0),
            Scalar::null(DType::Primitive(I32, Nullability::Nullable))
        );
    }
}
