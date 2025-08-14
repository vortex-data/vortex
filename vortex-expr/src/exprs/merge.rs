// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools as _;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray, ToCanonical};
use vortex_dtype::{DType, FieldNames, Nullability, StructFields};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};

use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

vtable!(Merge);

/// Merge zero or more expressions that ALL return structs.
///
/// If any field names are duplicated, the field from later expressions wins.
///
/// NOTE: Fields are not recursively merged, i.e. the later field REPLACES the earlier field.
/// This makes struct fields behaviour consistent with other dtypes.
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeExpr {
    values: Vec<ExprRef>,
    nullability: Nullability,
}

pub struct MergeExprEncoding;

impl VTable for MergeVTable {
    type Expr = MergeExpr;
    type Encoding = MergeExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("merge")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(MergeExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        expr.values.iter().collect()
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(MergeExpr {
            values: children,
            nullability: expr.nullability,
        })
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.is_empty() {
            vortex_bail!(
                "Merge expression must have at least one child, got: {:?}",
                children
            );
        }
        Ok(MergeExpr {
            values: children,
            nullability: Nullability::NonNullable, // Default to non-nullable
        })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let len = scope.len();
        let value_arrays = expr
            .values
            .iter()
            .map(|value_expr| value_expr.unchecked_evaluate(scope))
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

            let struct_array = value_array.to_struct()?;

            for (i, field_name) in struct_array.names().iter().enumerate() {
                let array = struct_array.fields()[i].clone();

                // Update or insert field.
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    arrays[idx] = array;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(array);
                }
            }
        }

        let validity = match expr.nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        Ok(
            StructArray::try_new(FieldNames::from(field_names), arrays, len, validity)?
                .into_array(),
        )
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();

        for value in expr.values.iter() {
            let dtype = value.return_dtype(scope)?;
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
            StructFields::new(FieldNames::from(field_names), arrays),
            expr.nullability,
        ))
    }
}

impl MergeExpr {
    pub fn new(values: Vec<ExprRef>, nullability: Nullability) -> Self {
        MergeExpr {
            values,
            nullability,
        }
    }

    pub fn new_expr(values: Vec<ExprRef>, nullability: Nullability) -> ExprRef {
        Self::new(values, nullability).into_expr()
    }

    pub fn nullability(&self) -> Nullability {
        self.nullability
    }
}

pub fn merge(
    elements: impl IntoIterator<Item = impl Into<ExprRef>>,
    nullability: Nullability,
) -> ExprRef {
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    MergeExpr::new(values, nullability).into_expr()
}

impl Display for MergeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "merge({}){}",
            self.values.iter().format(", "),
            self.nullability
        )
    }
}

impl AnalysisExpr for MergeExpr {}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_error::{VortexResult, vortex_bail};

    use crate::{MergeExpr, Scope, get_item, root};

    fn primitive_field(array: &dyn Array, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array.to_struct()?.field_by_name(field)?.clone();
        for field in field_path {
            array = array.to_struct()?.field_by_name(field)?.clone();
        }
        array.to_primitive()
    }

    #[test]
    pub fn test_merge() {
        let expr = MergeExpr::new(
            vec![
                get_item("0", root()),
                get_item("1", root()),
                get_item("2", root()),
            ],
            Nullability::NonNullable,
        );

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
        let actual_array = expr.evaluate(&Scope::new(test_array)).unwrap();

        assert_eq!(
            actual_array.as_struct_typed().names(),
            ["a", "b", "c", "d", "e"]
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
        let expr = MergeExpr::new(Vec::new(), Nullability::NonNullable);

        let test_array = StructArray::from_fields(&[("a", buffer![0, 1, 2].into_array())])
            .unwrap()
            .into_array();
        let actual_array = expr.evaluate(&Scope::new(test_array.clone())).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert_eq!(actual_array.as_struct_typed().nfields(), 0);
    }

    #[test]
    pub fn test_nested_merge() {
        // Nested structs are not merged!

        let expr = MergeExpr::new(
            vec![get_item("0", root()), get_item("1", root())],
            Nullability::NonNullable,
        );

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
        let actual_array = expr
            .evaluate(&Scope::new(test_array.clone()))
            .unwrap()
            .to_struct()
            .unwrap();

        assert_eq!(
            actual_array
                .field_by_name("a")
                .unwrap()
                .to_struct()
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
        let expr = MergeExpr::new(
            vec![get_item("0", root()), get_item("1", root())],
            Nullability::NonNullable,
        );

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
        let actual_array = expr
            .evaluate(&Scope::new(test_array.clone()))
            .unwrap()
            .to_struct()
            .unwrap();

        assert_eq!(actual_array.names(), ["a", "c", "b", "d"]);
    }

    #[test]
    pub fn test_merge_nullable() {
        let expr = MergeExpr::new(vec![get_item("0", root())], Nullability::Nullable);

        let test_array = StructArray::from_fields(&[(
            "0",
            StructArray::from_fields(&[
                ("a", buffer![0, 0, 0].into_array()),
                ("b", buffer![1, 1, 1].into_array()),
            ])
            .unwrap()
            .into_array(),
        )])
        .unwrap()
        .into_array();
        let actual_array = expr.evaluate(&Scope::new(test_array.clone())).unwrap();
        assert!(actual_array.dtype().is_nullable());
    }
}
