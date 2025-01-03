use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::field::Field;
use vortex_dtype::FieldNames;
use vortex_error::{vortex_bail, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone)]
pub struct Pack {
    names: FieldNames,
    values: Vec<ExprRef>,
    validity: Option<ExprRef>,
}

impl Pack {
    pub fn try_new_expr(
        names: FieldNames,
        values: Vec<ExprRef>,
        validity: Option<ExprRef>,
    ) -> VortexResult<Arc<Self>> {
        if names.len() != values.len() {
            vortex_bail!("length mismatch {} {}", names.len(), values.len());
        }
        if names.len() < 1 {
            vortex_bail!("must provide at least one field");
        }
        Ok(Arc::new(Pack {
            names,
            values,
            validity,
        }))
    }
}

impl PartialEq<dyn Any> for Pack {
    fn eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<Pack>().is_some_and(|other_pack| {
            self.names == other_pack.names
                && self
                    .values
                    .iter()
                    .zip(other_pack.values.iter())
                    .all(|(x, y)| x.eq(y))
        })
    }
}

impl Display for Pack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pack(")?;
        self.names
            .iter()
            .zip_eq(self.values.iter())
            .format_with(",", |(name, value), fmt| {
                fmt(&format_args!("{}: {}", name, value))
            });
        write!(f, ")")
    }
}

impl VortexExpr for Pack {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let value_arrays = self
            .values
            .iter()
            .map(|value_expr| value_expr.evaluate(batch))
            .process_results(|it| it.collect::<Vec<_>>())?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity_expr| validity_expr.evaluate(batch))
            .transpose()?
            .map_or_else(|| Validity::NonNullable, Validity::Array);
        let length = value_arrays[0].len();
        StructArray::try_new(self.names.clone(), value_arrays, length, validity)
            .map(IntoArrayData::into_array)
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        for expr in self.values.iter().chain(self.validity.iter()) {
            expr.collect_references(references);
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{PrimitiveArray, StructArray};
    use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant as _};
    use vortex_buffer::buffer;
    use vortex_dtype::field::Field;
    use vortex_dtype::FieldNames;
    use vortex_error::{vortex_bail, vortex_err, VortexResult};

    use crate::{Column, Pack, VortexExpr};

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
    }

    fn primitive_field(array: &ArrayData, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array
            .as_struct_array()
            .ok_or_else(|| vortex_err!("expected a struct"))?
            .field_by_name(field)
            .ok_or_else(|| vortex_err!("expected field to exist: {}", field))?;

        for field in field_path {
            array = array
                .as_struct_array()
                .ok_or_else(|| vortex_err!("expected a struct"))?
                .field_by_name(field)
                .ok_or_else(|| vortex_err!("expected field to exist: {}", field))?;
        }
        Ok(array.into_primitive().unwrap())
    }

    #[test]
    pub fn test_simple_pack() {
        let expr = Pack::try_new_expr(
            ["one".into(), "two".into(), "three".into()].into(),
            vec![
                Column::new_expr(Field::from("a")),
                Column::new_expr(Field::from("b")),
                Column::new_expr(Field::from("a")),
            ],
            None,
        )
        .unwrap();

        let actual_array = expr.evaluate(test_array().as_ref()).unwrap();
        let expected_names: FieldNames = ["one".into(), "two".into(), "three".into()].into();
        assert_eq!(
            actual_array.as_struct_array().unwrap().names(),
            &expected_names
        );

        assert_eq!(
            primitive_field(&actual_array, &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(&actual_array, &["two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(&actual_array, &["three"])
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
                Column::new_expr(Field::from("a")),
                Pack::try_new_expr(
                    ["two_one".into(), "two_two".into()].into(),
                    vec![
                        Column::new_expr(Field::from("b")),
                        Column::new_expr(Field::from("b")),
                    ],
                    None,
                )
                .unwrap(),
                Column::new_expr(Field::from("a")),
            ],
            None,
        )
        .unwrap();

        let actual_array = expr.evaluate(test_array().as_ref()).unwrap();
        let expected_names: FieldNames = ["one".into(), "two".into(), "three".into()].into();
        assert_eq!(
            actual_array.as_struct_array().unwrap().names(),
            &expected_names
        );

        assert_eq!(
            primitive_field(&actual_array, &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(&actual_array, &["two", "two_one"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(&actual_array, &["two", "two_two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(&actual_array, &["three"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
    }
}
