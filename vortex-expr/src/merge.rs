use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, ArrayVariants};
use vortex_dtype::{DType, FieldNames, Nullability, StructDType};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};

use crate::{ExprRef, VortexExpr};

/// Merge zero or more expressions that ALL return structs.
///
/// If any field names are duplicated, the field from later expressions wins.
///
/// NOTE: Fields are not recursively merged, i.e. the later field REPLACES the earlier field.
/// This makes struct fields behaviour consistent with other dtypes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Merge {
    values: Vec<ExprRef>,
}

impl Merge {
    pub fn new_expr(values: Vec<ExprRef>) -> Arc<Self> {
        Arc::new(Merge { values })
    }
}

pub fn merge(elements: impl IntoIterator<Item = impl Into<ExprRef>>) -> ExprRef {
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    Merge::new_expr(values)
}

impl Display for Merge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("{")?;
        self.values
            .iter()
            .format_with(", ", |expr, f| f(expr))
            .fmt(f)?;
        f.write_str("}")
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind::Kind;

    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id, Merge};

    pub struct MergeSerde;

    impl Id for MergeSerde {
        fn id(&self) -> &'static str {
            "merge"
        }
    }

    impl ExprDeserialize for MergeSerde {
        fn deserialize(&self, _kind: &Kind, _children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            vortex_bail!(NotImplemented: "", self.id())
        }
    }

    impl ExprSerializable for Merge {
        fn id(&self) -> &'static str {
            MergeSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            vortex_bail!(NotImplemented: "", self.id())
        }
    }
}

impl VortexExpr for Merge {
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

        // Collect fields in order of appearance. Later fields overwrite earlier fields.
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();

        for value_array in value_arrays.iter() {
            // TODO(marko): When nullable, we need to merge struct validity into field validity.
            if value_array.dtype().is_nullable() {
                todo!("merge nullable structs");
            }
            if !value_array.dtype().is_struct() {
                vortex_bail!("merge expects non-nullable struct input");
            }

            let struct_array = value_array
                .as_struct_typed()
                .vortex_expect("merge expects struct input");

            for (i, field_name) in struct_array.names().iter().enumerate() {
                let array = struct_array
                    .maybe_null_field_by_idx(i)
                    .vortex_expect("struct field not found");

                // Update or insert field.
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    arrays[idx] = array;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(array);
                }
            }
        }

        Ok(StructArray::try_new(
            FieldNames::from(field_names),
            arrays,
            len,
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn children(&self) -> Vec<&ExprRef> {
        self.values.iter().collect()
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        Self::new_expr(children)
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();

        for value in self.values.iter() {
            let dtype = value.return_dtype(scope_dtype)?;
            if !dtype.is_struct() {
                vortex_bail!("merge expects non-nullable struct input");
            }

            let struct_dtype = dtype
                .as_struct()
                .vortex_expect("merge expects struct input");

            for i in 0..struct_dtype.nfields() {
                let field_name = struct_dtype.field_name(i).vortex_expect("never OOB");
                let field_dtype = struct_dtype.field_by_index(i).vortex_expect("never OOB");
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    arrays[idx] = field_dtype;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(field_dtype);
                }
            }
        }

        Ok(DType::Struct(
            Arc::new(StructDType::new(FieldNames::from(field_names), arrays)),
            Nullability::NonNullable,
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_error::{VortexResult, vortex_bail, vortex_err};

    use crate::{GetItem, Identity, Merge, VortexExpr};

    fn primitive_field(array: &dyn Array, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array
            .as_struct_typed()
            .ok_or_else(|| vortex_err!("expected a struct"))?
            .maybe_null_field_by_name(field)?;

        for field in field_path {
            array = array
                .as_struct_typed()
                .ok_or_else(|| vortex_err!("expected a struct"))?
                .maybe_null_field_by_name(field)?;
        }
        Ok(array.to_primitive().unwrap())
    }

    #[test]
    pub fn test_merge() {
        let expr = Merge::new_expr(vec![
            GetItem::new_expr("0", Identity::new_expr()),
            GetItem::new_expr("1", Identity::new_expr()),
            GetItem::new_expr("2", Identity::new_expr()),
        ]);

        let test_array = StructArray::from_fields(&[
            (
                "0",
                StructArray::from_fields(&[
                    ("a", buffer![0, 0, 0].into_array()),
                    ("b", buffer![1, 1, 1].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
            (
                "1",
                StructArray::from_fields(&[
                    ("b", buffer![2, 2, 2].into_array()),
                    ("c", buffer![3, 3, 3].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
            (
                "2",
                StructArray::from_fields(&[
                    ("d", buffer![4, 4, 4].into_array()),
                    ("e", buffer![5, 5, 5].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array();
        let actual_array = expr.evaluate(test_array.as_ref()).unwrap();

        assert_eq!(
            actual_array.as_struct_typed().unwrap().names(),
            &["a".into(), "b".into(), "c".into(), "d".into(), "e".into()].into()
        );

        assert_eq!(
            primitive_field(&actual_array, &["a"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 0, 0]
        );
        assert_eq!(
            primitive_field(&actual_array, &["b"])
                .unwrap()
                .as_slice::<i32>(),
            [2, 2, 2]
        );
        assert_eq!(
            primitive_field(&actual_array, &["c"])
                .unwrap()
                .as_slice::<i32>(),
            [3, 3, 3]
        );
        assert_eq!(
            primitive_field(&actual_array, &["d"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 4, 4]
        );
        assert_eq!(
            primitive_field(&actual_array, &["e"])
                .unwrap()
                .as_slice::<i32>(),
            [5, 5, 5]
        );
    }

    #[test]
    pub fn test_empty_merge() {
        let expr = Merge::new_expr(Vec::new());

        let test_array = StructArray::from_fields(&[("a", buffer![0, 1, 2].into_array())])
            .unwrap()
            .into_array();
        let actual_array = expr.evaluate(&test_array).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert_eq!(actual_array.as_struct_typed().unwrap().nfields(), 0);
    }

    #[test]
    pub fn test_nested_merge() {
        // Nested structs are not merged!

        let expr = Merge::new_expr(vec![
            GetItem::new_expr("0", Identity::new_expr()),
            GetItem::new_expr("1", Identity::new_expr()),
        ]);

        let test_array = StructArray::from_fields(&[
            (
                "0",
                StructArray::from_fields(&[(
                    "a",
                    StructArray::from_fields(&[
                        ("x", buffer![0, 0, 0].into_array()),
                        ("y", buffer![1, 1, 1].into_array()),
                    ])
                    .unwrap()
                    .into_array(),
                )])
                .unwrap()
                .into_array(),
            ),
            (
                "1",
                StructArray::from_fields(&[(
                    "a",
                    StructArray::from_fields(&[("x", buffer![0, 0, 0].into_array())])
                        .unwrap()
                        .into_array(),
                )])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array();
        let actual_array = expr.evaluate(test_array.as_ref()).unwrap();

        assert_eq!(
            actual_array
                .as_struct_typed()
                .unwrap()
                .maybe_null_field_by_name("a")
                .unwrap()
                .as_struct_typed()
                .unwrap()
                .names()
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<_>>(),
            vec!["x"]
        );
    }

    #[test]
    pub fn test_merge_order() {
        let expr = Merge::new_expr(vec![
            GetItem::new_expr("0", Identity::new_expr()),
            GetItem::new_expr("1", Identity::new_expr()),
        ]);

        let test_array = StructArray::from_fields(&[
            (
                "0",
                StructArray::from_fields(&[
                    ("a", buffer![0, 0, 0].into_array()),
                    ("c", buffer![1, 1, 1].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
            (
                "1",
                StructArray::from_fields(&[
                    ("b", buffer![2, 2, 2].into_array()),
                    ("d", buffer![3, 3, 3].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array();
        let actual_array = expr.evaluate(test_array.as_ref()).unwrap();

        assert_eq!(
            actual_array.as_struct_typed().unwrap().names(),
            &["a".into(), "c".into(), "b".into(), "d".into()].into()
        );
    }
}
