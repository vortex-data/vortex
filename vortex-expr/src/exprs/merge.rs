// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools as _;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{
    Array, ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray as _, ToCanonical,
};
use vortex_dtype::{DType, FieldNames, Nullability, StructFields};
use vortex_error::{VortexResult, vortex_bail};
use vortex_utils::aliases::hash_set::HashSet;

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
    duplicate_handling: DuplicateHandling,
}

impl MergeExpr {
    pub fn duplicate_handling(&self) -> DuplicateHandling {
        self.duplicate_handling
    }
}

/// What to do when merged structs share a field name.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DuplicateHandling {
    /// If two structs share a field name, take the value from the right-most struct.
    RightMost,
    /// If two structs share a field name, error.
    #[default]
    Error,
}

impl Display for DuplicateHandling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DuplicateHandling::Error => write!(f, "error"),
            DuplicateHandling::RightMost => write!(f, "right-most"),
        }
    }
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
            duplicate_handling: expr.duplicate_handling,
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
            duplicate_handling: DuplicateHandling::default(),
        })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        // Collect fields in order of appearance. Later fields overwrite earlier fields.
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();
        let mut duplicate_names = HashSet::<_>::new();

        for expr in expr.values.iter() {
            // TODO(marko): When nullable, we need to merge struct validity into field validity.
            let array = expr.unchecked_evaluate(scope)?;
            if array.dtype().is_nullable() {
                vortex_bail!("merge expects non-nullable input");
            }
            if !array.dtype().is_struct() {
                vortex_bail!("merge expects struct input");
            }
            let array = array.to_struct();

            for (field_name, array) in array.names().iter().zip_eq(array.fields().iter().cloned()) {
                // Update or insert field.
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    duplicate_names.insert(field_name.clone());
                    arrays[idx] = array;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(array);
                }
            }
        }

        if expr.duplicate_handling == DuplicateHandling::Error && !duplicate_names.is_empty() {
            vortex_bail!(
                "merge: duplicate fields in children: {}",
                duplicate_names.into_iter().format(", ")
            )
        }

        // TODO(DK): When children are allowed to be nullable, this needs to change.
        let validity = Validity::NonNullable;
        let len = scope.len();
        Ok(
            StructArray::try_new(FieldNames::from(field_names), arrays, len, validity)?
                .into_array(),
        )
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();
        let mut merge_nullability = Nullability::NonNullable;
        let mut duplicate_names = HashSet::<_>::new();

        for expr in expr.values.iter() {
            let dtype = expr.return_dtype(scope)?;
            let Some(fields) = dtype.as_struct_fields_opt() else {
                vortex_bail!("merge expects struct input");
            };
            if dtype.is_nullable() {
                vortex_bail!("merge expects non-nullable input");
            }

            merge_nullability |= dtype.nullability();

            for (field_name, field_dtype) in fields.names().iter().zip_eq(fields.fields()) {
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    duplicate_names.insert(field_name.clone());
                    arrays[idx] = field_dtype;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(field_dtype);
                }
            }
        }

        if expr.duplicate_handling == DuplicateHandling::Error && !duplicate_names.is_empty() {
            vortex_bail!(
                "merge: duplicate fields in children: {}",
                duplicate_names.into_iter().format(", ")
            )
        }

        Ok(DType::Struct(
            StructFields::new(FieldNames::from(field_names), arrays),
            merge_nullability,
        ))
    }
}

impl MergeExpr {
    pub fn new(values: Vec<ExprRef>) -> Self {
        MergeExpr {
            values,
            duplicate_handling: DuplicateHandling::default(),
        }
    }

    pub fn new_expr(values: Vec<ExprRef>) -> ExprRef {
        Self::new(values).into_expr()
    }

    pub fn new_opts(values: Vec<ExprRef>, duplicate_handling: DuplicateHandling) -> Self {
        MergeExpr {
            values,
            duplicate_handling,
        }
    }

    pub fn new_expr_opts(values: Vec<ExprRef>, duplicate_handling: DuplicateHandling) -> ExprRef {
        Self::new_opts(values, duplicate_handling).into_expr()
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

pub fn merge_opts(
    elements: impl IntoIterator<Item = impl Into<ExprRef>>,
    duplicate_handling: DuplicateHandling,
) -> ExprRef {
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    MergeExpr::new_opts(values, duplicate_handling).into_expr()
}

impl DisplayAs for MergeExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(
                    f,
                    "merge[{}]({})",
                    self.duplicate_handling,
                    self.values.iter().format(", "),
                )
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

    use crate::{DuplicateHandling, MergeExpr, Scope, get_item, merge, root};

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
    pub fn test_merge_right_most() {
        let expr = MergeExpr::new_opts(
            vec![
                get_item("0", root()),
                get_item("1", root()),
                get_item("2", root()),
            ],
            DuplicateHandling::RightMost,
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
    #[should_panic(expected = "merge: duplicate fields in children")]
    pub fn test_merge_error_on_dupe_return_dtype() {
        let expr = MergeExpr::new_opts(
            vec![get_item("0", root()), get_item("1", root())],
            DuplicateHandling::Error,
        );
        let test_array = StructArray::try_from_iter([
            (
                "0",
                StructArray::try_from_iter([("a", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
            (
                "1",
                StructArray::try_from_iter([("c", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
        ])
        .unwrap()
        .into_array();

        expr.return_dtype(test_array.dtype()).unwrap();
    }

    #[test]
    #[should_panic(expected = "merge: duplicate fields in children")]
    pub fn test_merge_error_on_dupe_evaluate() {
        let expr = MergeExpr::new_opts(
            vec![get_item("0", root()), get_item("1", root())],
            DuplicateHandling::Error,
        );
        let test_array = StructArray::try_from_iter([
            (
                "0",
                StructArray::try_from_iter([("a", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
            (
                "1",
                StructArray::try_from_iter([("c", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
        ])
        .unwrap()
        .into_array();

        expr.evaluate(&Scope::new(test_array)).unwrap();
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

        let expr = MergeExpr::new_opts(
            vec![get_item("0", root()), get_item("1", root())],
            DuplicateHandling::RightMost,
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
        assert_eq!(expr.to_string(), "merge[error]($.struct1, $.struct2)");

        let expr2 = MergeExpr::new(vec![get_item("a", root())]);
        assert_eq!(expr2.to_string(), "merge[error]($.a)");
    }
}
