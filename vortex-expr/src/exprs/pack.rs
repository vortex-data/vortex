// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::hash::Hash;

use itertools::Itertools as _;
use prost::Message;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructFields};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;

use crate::{ChildName, ExprId, Expression, ExpressionView, VTable, VTableExt};

/// Pack zero or more expressions into a structure with named fields.
pub struct Pack;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackOptions {
    pub names: FieldNames,
    pub nullability: Nullability,
}

impl VTable for Pack {
    type Instance = PackOptions;

    fn id(&self) -> ExprId {
        ExprId::new_ref("vortex.pack")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::PackOpts {
                paths: instance.names.iter().map(|n| n.to_string()).collect(),
                nullable: instance.nullability.into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        let opts = pb::PackOpts::decode(metadata)?;
        let names: FieldNames = opts
            .paths
            .iter()
            .map(|name| FieldName::from(name.as_str()))
            .collect();
        Ok(Some(PackOptions {
            names,
            nullability: opts.nullable.into(),
        }))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        let instance = expr.data();
        if expr.children().len() != instance.names.len() {
            vortex_bail!(
                "Pack expression expects {} children, got {}",
                instance.names.len(),
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, instance: &Self::Instance, child_idx: usize) -> ChildName {
        match instance.names.get(child_idx) {
            Some(name) => ChildName::from(name.inner().clone()),
            None => unreachable!(
                "Invalid child index {} for Pack expression with {} fields",
                child_idx,
                instance.names.len()
            ),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "pack(")?;
        for (i, (name, child)) in expr
            .data()
            .names
            .iter()
            .zip(expr.children().iter())
            .enumerate()
        {
            write!(f, "{}: ", name)?;
            child.fmt_sql(f)?;
            if i + 1 < expr.data().names.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, "){}", expr.data().nullability)
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let value_dtypes = expr
            .children()
            .iter()
            .map(|child| child.return_dtype(scope))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(DType::Struct(
            StructFields::new(expr.data().names.clone(), value_dtypes),
            expr.data().nullability,
        ))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let len = scope.len();
        let value_arrays = expr
            .children()
            .iter()
            .zip_eq(expr.data().names.iter())
            .map(|(child_expr, name)| {
                child_expr
                    .evaluate(scope)
                    .map_err(|e| e.with_context(format!("Can't evaluate '{name}'")))
            })
            .process_results(|it| it.collect::<Vec<_>>())?;
        let validity = match expr.data().nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        Ok(
            StructArray::try_new(expr.data().names.clone(), value_arrays, len, validity)?
                .into_array(),
        )
    }
}

impl ExpressionView<'_, Pack> {
    pub fn field(&self, field_name: &FieldName) -> VortexResult<Expression> {
        let idx = self
            .data()
            .names
            .iter()
            .position(|name| name == field_name)
            .ok_or_else(|| {
                vortex_err!(
                    "Cannot find field {} in pack fields {:?}",
                    field_name,
                    self.data().names
                )
            })?;

        Ok(self.child(idx).clone())
    }
}

/// Creates an expression that packs values into a struct with named fields.
///
/// ```rust
/// # use vortex_dtype::Nullability;
/// # use vortex_expr::{pack, col, lit};
/// let expr = pack([("id", col("user_id")), ("constant", lit(42))], Nullability::NonNullable);
/// ```
pub fn pack(
    elements: impl IntoIterator<Item = (impl Into<FieldName>, Expression)>,
    nullability: Nullability,
) -> Expression {
    let (names, values): (Vec<_>, Vec<_>) = elements
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .unzip();
    Pack.new_expr(
        PackOptions {
            names: names.into(),
            nullability,
        },
        values,
    )
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_error::{VortexResult, vortex_bail};

    use super::{Pack, PackOptions, pack};
    use crate::exprs::get_item::col;
    use crate::{Scope, VTableExt};

    fn test_array() -> ArrayRef {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array()
    }

    fn primitive_field(array: &dyn Array, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array.to_struct().field_by_name(field)?.clone();
        for field in field_path {
            array = array.to_struct().field_by_name(field)?.clone();
        }
        Ok(array.to_primitive())
    }

    #[test]
    pub fn test_empty_pack() {
        let expr = Pack.new_expr(
            PackOptions {
                names: Default::default(),
                nullability: Default::default(),
            },
            [],
        );

        let test_array = test_array();
        let actual_array = expr.evaluate(&Scope::new(test_array.clone())).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert_eq!(actual_array.to_struct().struct_fields().nfields(), 0);
    }

    #[test]
    pub fn test_simple_pack() {
        let expr = Pack.new_expr(
            PackOptions {
                names: ["one", "two", "three"].into(),
                nullability: Nullability::NonNullable,
            },
            [col("a"), col("b"), col("a")],
        );

        let actual_array = expr
            .evaluate(&Scope::new(test_array()))
            .unwrap()
            .to_struct();

        assert_eq!(actual_array.names(), ["one", "two", "three"]);
        assert_eq!(actual_array.validity(), &Validity::NonNullable);

        assert_eq!(
            primitive_field(actual_array.as_ref(), &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["three"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
    }

    #[test]
    pub fn test_nested_pack() {
        let expr = Pack.new_expr(
            PackOptions {
                names: ["one", "two", "three"].into(),
                nullability: Nullability::NonNullable,
            },
            [
                col("a"),
                Pack.new_expr(
                    PackOptions {
                        names: ["two_one", "two_two"].into(),
                        nullability: Nullability::NonNullable,
                    },
                    [col("b"), col("b")],
                ),
                col("a"),
            ],
        );

        let actual_array = expr
            .evaluate(&Scope::new(test_array()))
            .unwrap()
            .to_struct();

        assert_eq!(actual_array.names(), ["one", "two", "three"]);

        assert_eq!(
            primitive_field(actual_array.as_ref(), &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["two", "two_one"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["two", "two_two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["three"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
    }

    #[test]
    pub fn test_pack_nullable() {
        let expr = Pack.new_expr(
            PackOptions {
                names: ["one", "two", "three"].into(),
                nullability: Nullability::Nullable,
            },
            [col("a"), col("b"), col("a")],
        );

        let actual_array = expr
            .evaluate(&Scope::new(test_array()))
            .unwrap()
            .to_struct();

        assert_eq!(actual_array.names(), ["one", "two", "three"]);
        assert_eq!(actual_array.validity(), &Validity::AllValid);
    }

    #[test]
    pub fn test_display() {
        let expr = pack(
            [("id", col("user_id")), ("name", col("username"))],
            Nullability::NonNullable,
        );
        assert_eq!(expr.to_string(), "pack(id: $.user_id, name: $.username)");

        let expr2 = Pack.new_expr(
            PackOptions {
                names: ["x", "y"].into(),
                nullability: Nullability::Nullable,
            },
            [col("a"), col("b")],
        );
        assert_eq!(expr2.to_string(), "pack(x: $.a, y: $.b)?");
    }
}
