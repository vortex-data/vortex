// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use itertools::Itertools as _;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray, ToCanonical};
use vortex_dtype::{DType, FieldNames, Nullability, StructFields};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};

use crate::display::{DisplayAs, DisplayFormat};
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

    fn with_children(_expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(MergeExpr { values: children })
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
        Ok(MergeExpr { values: children })
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

            let struct_array = value_array.to_struct();

            for (field_name, array) in struct_array
                .names()
                .iter()
                .zip_eq(struct_array.fields().iter().cloned())
            {
                // Update or insert field.
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    arrays[idx] = array;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(array);
                }
            }
        }

        // TODO(DK): When children are allowed to be nullable, this needs to change.
        let validity = Validity::NonNullable;
        Ok(
            StructArray::try_new(FieldNames::from(field_names), arrays, len, validity)?
                .into_array(),
        )
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();

        let mut nullability = Nullability::NonNullable;

        for value in expr.values.iter() {
            let dtype = value.return_dtype(scope)?;
            if !dtype.is_struct() {
                vortex_bail!("merge expects struct input");
            }
            if dtype.is_nullable() {
                vortex_bail!("merge expects non-nullable input");
            }
            nullability |= dtype.nullability();

            let struct_dtype = dtype
                .as_struct_fields_opt()
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
            nullability,
        ))
    }
}

impl MergeExpr {
    pub fn new(values: Vec<ExprRef>) -> Self {
        MergeExpr { values }
    }

    pub fn new_expr(values: Vec<ExprRef>) -> ExprRef {
        Self::new(values).into_expr()
    }
}

/// Creates an expression that merges struct expressions into a single struct.
///
/// Combines fields from all input expressions. If field names are duplicated,
/// later expressions win. Fields are not recursively merged.
///
/// ```rust
/// # use vortex_dtype::Nullability;
/// # use vortex_expr::{merge, get_item, root};
/// let expr = merge([get_item("a", root()), get_item("b", root())]);
/// ```
pub fn merge(elements: impl IntoIterator<Item = impl Into<ExprRef>>) -> ExprRef {
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    MergeExpr::new(values).into_expr()
}

impl DisplayAs for MergeExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "merge({})", self.values.iter().format(", "),)
            }
            DisplayFormat::Tree => {
                write!(f, "Merge")
            }
        }
    }
}

impl AnalysisExpr for MergeExpr {}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_error::{VortexResult, vortex_bail};

    use crate::{MergeExpr, Scope, get_item, merge, root};

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
    pub fn test_merge() {
        let expr = MergeExpr::new(vec![
            get_item("0", root()),
            get_item("1", root()),
            get_item("2", root()),
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
        let expr = MergeExpr::new(Vec::new());

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

        let expr = MergeExpr::new(vec![get_item("0", root()), get_item("1", root())]);

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
            .to_struct();

        assert_eq!(
            actual_array
                .field_by_name("a")
                .unwrap()
                .to_struct()
                .names()
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<_>>(),
            vec!["x"]
        );
    }

    #[test]
    pub fn test_merge_order() {
        let expr = MergeExpr::new(vec![get_item("0", root()), get_item("1", root())]);

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
            .to_struct();

        assert_eq!(actual_array.names(), ["a", "c", "b", "d"]);
    }

    #[test]
    pub fn test_display() {
        let expr = merge([get_item("struct1", root()), get_item("struct2", root())]);
        assert_eq!(expr.to_string(), "merge($.struct1, $.struct2)");

        let expr2 = MergeExpr::new(vec![get_item("a", root())]);
        assert_eq!(expr2.to_string(), "merge($.a)");
    }
}
