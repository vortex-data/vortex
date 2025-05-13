use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructDType};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{ExprRef, VortexExpr};

/// Pack zero or more expressions into a structure with named fields.
///
/// # Examples
///
/// ```
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_buffer::buffer;
/// use vortex_expr::{Pack, Identity, VortexExpr};
/// use vortex_scalar::Scalar;
/// use vortex_dtype::Nullability;
///
/// let example = Pack::try_new_expr(
///     ["x".into(), "x copy".into(), "second x copy".into()].into(),
///     vec![Identity::new_expr(), Identity::new_expr(), Identity::new_expr()],
///     Nullability::NonNullable,
/// ).unwrap();
/// let packed = example.evaluate(&buffer![100, 110, 200].into_array()).unwrap();
/// let x_copy = packed
///     .to_struct()
///     .unwrap()
///     .field_by_name("x copy")
///     .unwrap()
///     .clone();
/// assert_eq!(x_copy.scalar_at(0).unwrap(), Scalar::from(100));
/// assert_eq!(x_copy.scalar_at(1).unwrap(), Scalar::from(110));
/// assert_eq!(x_copy.scalar_at(2).unwrap(), Scalar::from(200));
/// ```
///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pack {
    names: FieldNames,
    values: Vec<ExprRef>,
    nullability: Nullability,
}

impl Pack {
    pub fn try_new_expr(
        names: FieldNames,
        values: Vec<ExprRef>,
        nullability: Nullability,
    ) -> VortexResult<Arc<Self>> {
        if names.len() != values.len() {
            vortex_bail!("length mismatch {} {}", names.len(), values.len());
        }
        Ok(Arc::new(Pack {
            names,
            values,
            nullability,
        }))
    }

    pub fn names(&self) -> &FieldNames {
        &self.names
    }

    pub fn field(&self, field_name: &FieldName) -> VortexResult<ExprRef> {
        let idx = self
            .names
            .iter()
            .position(|name| name == field_name)
            .ok_or_else(|| {
                vortex_err!(
                    "Cannot find field {} in pack fields {:?}",
                    field_name,
                    self.names
                )
            })?;

        self.values
            .get(idx)
            .cloned()
            .ok_or_else(|| vortex_err!("field index out of bounds: {}", idx))
    }
}

pub fn pack(
    elements: impl IntoIterator<Item = (impl Into<FieldName>, ExprRef)>,
    nullability: Nullability,
) -> ExprRef {
    let (names, values): (Vec<_>, Vec<_>) = elements
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .unzip();
    Pack::try_new_expr(names.into(), values, nullability)
        .vortex_expect("pack names and values have the same length")
}

impl Display for Pack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("{")?;
        self.names
            .iter()
            .zip(&self.values)
            .format_with(", ", |(name, expr), f| f(&format_args!("{name}: {expr}")))
            .fmt(f)?;
        f.write_str("}")
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind::Kind;

    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id, Pack};

    pub struct PackSerde;

    impl Id for PackSerde {
        fn id(&self) -> &'static str {
            "pack"
        }
    }

    impl ExprDeserialize for PackSerde {
        fn deserialize(&self, _kind: &Kind, _children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            todo!()
        }
    }

    impl ExprSerializable for Pack {
        fn id(&self) -> &'static str {
            PackSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            vortex_bail!(NotImplemented: "", self.id())
        }
    }
}

impl VortexExpr for Pack {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        let len = batch.len();
        let value_arrays = self
            .values
            .iter()
            .map(|value_expr| value_expr.evaluate(batch))
            .process_results(|it| it.collect::<Vec<_>>())?;
        let validity = match self.nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        Ok(StructArray::try_new(self.names.clone(), value_arrays, len, validity)?.into_array())
    }

    fn children(&self) -> Vec<&ExprRef> {
        self.values.iter().collect()
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), self.values.len());
        Self::try_new_expr(self.names.clone(), children, self.nullability)
            .vortex_expect("children are known to have the same length as names")
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let value_dtypes = self
            .values
            .iter()
            .map(|value_expr| value_expr.return_dtype(scope_dtype))
            .process_results(|it| it.collect())?;
        Ok(DType::Struct(
            Arc::new(StructDType::new(self.names.clone(), value_dtypes)),
            self.nullability,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{FieldNames, Nullability};
    use vortex_error::{VortexResult, vortex_bail};

    use crate::{Pack, VortexExpr, col};

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
    }

    fn primitive_field(array: &dyn Array, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array.to_struct()?.field_by_name(field)?.clone();
        for field in field_path {
            array = array.to_struct()?.field_by_name(field)?.clone();
        }
        Ok(array.to_primitive().unwrap())
    }

    #[test]
    pub fn test_empty_pack() {
        let expr = Pack::try_new_expr(Arc::new([]), Vec::new(), Nullability::NonNullable).unwrap();

        let test_array = test_array().into_array();
        let actual_array = expr.evaluate(&test_array).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert_eq!(
            actual_array.to_struct().unwrap().struct_dtype().nfields(),
            0
        );
    }

    #[test]
    pub fn test_simple_pack() {
        let expr = Pack::try_new_expr(
            ["one".into(), "two".into(), "three".into()].into(),
            vec![col("a"), col("b"), col("a")],
            Nullability::NonNullable,
        )
        .unwrap();

        let actual_array = expr
            .evaluate(test_array().as_ref())
            .unwrap()
            .to_struct()
            .unwrap();
        let expected_names: FieldNames = ["one".into(), "two".into(), "three".into()].into();
        assert_eq!(actual_array.names(), &expected_names);
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
        let expr = Pack::try_new_expr(
            ["one".into(), "two".into(), "three".into()].into(),
            vec![
                col("a"),
                Pack::try_new_expr(
                    ["two_one".into(), "two_two".into()].into(),
                    vec![col("b"), col("b")],
                    Nullability::NonNullable,
                )
                .unwrap(),
                col("a"),
            ],
            Nullability::NonNullable,
        )
        .unwrap();

        let actual_array = expr
            .evaluate(test_array().as_ref())
            .unwrap()
            .to_struct()
            .unwrap();
        let expected_names: FieldNames = ["one".into(), "two".into(), "three".into()].into();
        assert_eq!(actual_array.names(), &expected_names);

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
        let expr = Pack::try_new_expr(
            ["one".into(), "two".into(), "three".into()].into(),
            vec![col("a"), col("b"), col("a")],
            Nullability::Nullable,
        )
        .unwrap();

        let actual_array = expr
            .evaluate(test_array().as_ref())
            .unwrap()
            .to_struct()
            .unwrap();
        let expected_names: FieldNames = ["one".into(), "two".into(), "three".into()].into();
        assert_eq!(actual_array.names(), &expected_names);
        assert_eq!(actual_array.validity(), &Validity::AllValid);
    }
}
