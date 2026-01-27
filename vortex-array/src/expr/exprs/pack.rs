// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use itertools::Itertools as _;
use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_dtype::StructFields;
use vortex_error::VortexResult;
use vortex_proto::expr as pb;

use crate::IntoArray;
use crate::arrays::StructArray;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::lit;
use crate::validity::Validity;

/// Pack zero or more expressions into a structure with named fields.
pub struct Pack;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackOptions {
    pub names: FieldNames,
    pub nullability: Nullability,
}

impl Display for PackOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "names: [{}], nullability: {:#}",
            self.names.iter().join(", "),
            self.nullability
        )
    }
}

impl VTable for Pack {
    type Options = PackOptions;

    fn id(&self) -> ExprId {
        ExprId::new_ref("vortex.pack")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::PackOpts {
                paths: instance.names.iter().map(|n| n.to_string()).collect(),
                nullable: instance.nullability.into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
        let opts = pb::PackOpts::decode(metadata)?;
        let names: FieldNames = opts
            .paths
            .iter()
            .map(|name| FieldName::from(name.as_str()))
            .collect();
        Ok(PackOptions {
            names,
            nullability: opts.nullable.into(),
        })
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(options.names.len())
    }

    fn child_name(&self, instance: &Self::Options, child_idx: usize) -> ChildName {
        match instance.names.get(child_idx) {
            Some(name) => ChildName::from(name.inner().clone()),
            None => unreachable!(
                "Invalid child index {} for Pack expression with {} fields",
                child_idx,
                instance.names.len()
            ),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "pack(")?;
        for (i, (name, child)) in options.names.iter().zip(expr.children().iter()).enumerate() {
            write!(f, "{}: ", name)?;
            child.fmt_sql(f)?;
            if i + 1 < options.names.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, "){}", options.nullability)
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Struct(
            StructFields::new(options.names.clone(), arg_dtypes.to_vec()),
            options.nullability,
        ))
    }

    fn validity(
        &self,
        _options: &Self::Options,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(lit(true)))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: ExecutionArgs,
    ) -> VortexResult<ExecutionResult> {
        let len = args.row_count;
        let value_arrays = args.inputs;
        let validity: Validity = options.nullability.into();
        StructArray::try_new(options.names.clone(), value_arrays, len, validity)?
            .into_array()
            .execute(args.ctx)
    }

    // This applies a nullability
    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _instance: &Self::Options) -> bool {
        false
    }
}

/// Creates an expression that packs values into a struct with named fields.
///
/// ```rust
/// # use vortex_dtype::Nullability;
/// # use vortex_array::expr::{pack, col, lit};
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
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::Pack;
    use super::PackOptions;
    use super::pack;
    use crate::Array;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::expr::VTableExt;
    use crate::expr::exprs::get_item::col;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;

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

        let mut array = array.to_struct().unmasked_field_by_name(field)?.clone();
        for field in field_path {
            array = array.to_struct().unmasked_field_by_name(field)?.clone();
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
        let actual_array = test_array.clone().apply(&expr).unwrap();
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

        let actual_array = test_array().apply(&expr).unwrap().to_struct();

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

        let actual_array = test_array().apply(&expr).unwrap().to_struct();

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

        let actual_array = test_array().apply(&expr).unwrap().to_struct();

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
