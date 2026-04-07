// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use itertools::Itertools as _;
use prost::Message;
use vortex_error::VortexResult;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::lit;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::validity::Validity;

/// Pack zero or more expressions into a structure with named fields.
#[derive(Clone)]
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

impl ScalarFnVTable for Pack {
    type Options = PackOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.pack")
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

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::PackOpts::decode(_metadata)?;
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
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let len = args.row_count();
        let value_arrays: Vec<ArrayRef> = (0..args.num_inputs())
            .map(|i| args.get(i))
            .collect::<VortexResult<_>>()?;
        let validity: Validity = options.nullability.into();
        StructArray::try_new(options.names.clone(), value_arrays, len, validity)?
            .into_array()
            .execute(ctx)
    }

    // This applies a nullability
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
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::Pack;
    use super::PackOptions;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::struct_::StructArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::expr::col;
    use crate::expr::pack;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::pack::StructArray;
    use crate::validity::Validity;

    fn test_array() -> ArrayRef {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array()
    }

    fn primitive_field(array: &ArrayRef, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
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
        assert!(matches!(actual_array.validity(), Ok(Validity::NonNullable)));

        assert_arrays_eq!(
            primitive_field(&actual_array.clone().into_array(), &["one"]).unwrap(),
            PrimitiveArray::from_iter([0i32, 1, 2])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array.clone().into_array(), &["two"]).unwrap(),
            PrimitiveArray::from_iter([4i32, 5, 6])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array.into_array(), &["three"]).unwrap(),
            PrimitiveArray::from_iter([0i32, 1, 2])
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

        assert_arrays_eq!(
            primitive_field(&actual_array.clone().into_array(), &["one"]).unwrap(),
            PrimitiveArray::from_iter([0i32, 1, 2])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array.clone().into_array(), &["two", "two_one"]).unwrap(),
            PrimitiveArray::from_iter([4i32, 5, 6])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array.clone().into_array(), &["two", "two_two"]).unwrap(),
            PrimitiveArray::from_iter([4i32, 5, 6])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array.into_array(), &["three"]).unwrap(),
            PrimitiveArray::from_iter([0i32, 1, 2])
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
        assert!(matches!(actual_array.validity(), Ok(Validity::AllValid)));
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
