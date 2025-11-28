// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod transform;

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;
use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::FieldNames;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_proto::expr::select_opts::Opts;
use vortex_proto::expr::FieldNames as ProtoFieldNames;
use vortex_proto::expr::SelectOpts;
use vortex_vector::struct_::StructVector;
use vortex_vector::Vector;

use crate::expr::expression::Expression;
use crate::expr::field::DisplayFieldNames;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::ExpressionView;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldSelection {
    Include(FieldNames),
    Exclude(FieldNames),
}

pub struct Select;

impl VTable for Select {
    type Instance = FieldSelection;

    fn id(&self) -> ExprId {
        ExprId::new_ref("vortex.select")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        let opts = match instance {
            FieldSelection::Include(fields) => Opts::Include(ProtoFieldNames {
                names: fields.iter().map(|f| f.to_string()).collect(),
            }),
            FieldSelection::Exclude(fields) => Opts::Exclude(ProtoFieldNames {
                names: fields.iter().map(|f| f.to_string()).collect(),
            }),
        };

        let select_opts = SelectOpts { opts: Some(opts) };
        Ok(Some(select_opts.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        let prost_metadata = SelectOpts::decode(metadata)?;

        let select_opts = prost_metadata
            .opts
            .ok_or_else(|| vortex_err!("SelectOpts missing opts field"))?;

        let field_selection = match select_opts {
            Opts::Include(field_names) => FieldSelection::Include(FieldNames::from_iter(
                field_names.names.iter().map(|s| s.as_str()),
            )),
            Opts::Exclude(field_names) => FieldSelection::Exclude(FieldNames::from_iter(
                field_names.names.iter().map(|s| s.as_str()),
            )),
        };

        Ok(Some(field_selection))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 1 {
            vortex_bail!(
                "Select expression requires exactly 1 child, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::new_ref("child"),
            _ => unreachable!(),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        expr.child().fmt_sql(f)?;
        match expr.data() {
            FieldSelection::Include(fields) => {
                write!(f, "{{{}}}", DisplayFieldNames(fields))
            }
            FieldSelection::Exclude(fields) => {
                write!(f, "{{~ {}}}", DisplayFieldNames(fields))
            }
        }
    }

    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> std::fmt::Result {
        let names = match instance {
            FieldSelection::Include(names) => {
                write!(f, "include=")?;
                names
            }
            FieldSelection::Exclude(names) => {
                write!(f, "exclude=")?;
                names
            }
        };
        write!(f, "{{{}}}", DisplayFieldNames(names))
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let child_dtype = expr.child().return_dtype(scope)?;
        let child_struct_dtype = child_dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Select child not a struct dtype"))?;

        let projected = match expr.data() {
            FieldSelection::Include(fields) => child_struct_dtype.project(fields.as_ref())?,
            FieldSelection::Exclude(fields) => child_struct_dtype
                .names()
                .iter()
                .cloned()
                .zip_eq(child_struct_dtype.fields())
                .filter(|(name, _)| !fields.as_ref().contains(name))
                .collect(),
        };

        Ok(DType::Struct(projected, child_dtype.nullability()))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let batch = expr.child().evaluate(scope)?.to_struct();
        Ok(match expr.data() {
            FieldSelection::Include(f) => batch.project(f.as_ref()),
            FieldSelection::Exclude(names) => {
                let included_names = batch
                    .names()
                    .iter()
                    .filter(|&f| !names.as_ref().contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                batch.project(included_names.as_slice())
            }
        }?
        .into_array())
    }

    fn execute(&self, selection: &FieldSelection, mut args: ExecutionArgs) -> VortexResult<Vector> {
        let child = args
            .vectors
            .pop()
            .vortex_expect("Missing input child")
            .into_struct();
        let child_fields = args
            .dtypes
            .pop()
            .vortex_expect("Missing input dtype")
            .into_struct_fields();

        let field_indices: Vec<usize> = match selection {
            FieldSelection::Include(f) => f
                .iter()
                .map(|name| {
                    child_fields
                        .find(name)
                        .ok_or_else(|| vortex_err!("Field {} not found in struct dtype", name))
                })
                .try_collect(),
            FieldSelection::Exclude(names) => child_fields
                .names()
                .iter()
                .filter(|&f| !names.as_ref().contains(f))
                .map(|name| {
                    child_fields
                        .find(name)
                        .ok_or_else(|| vortex_err!("Field {} not found in struct dtype", name))
                })
                .try_collect(),
        }?;

        let (fields, mask) = child.into_parts();
        let new_fields = field_indices
            .iter()
            .map(|&idx| fields[idx].clone())
            .collect();
        Ok(unsafe { StructVector::new_unchecked(Arc::new(new_fields), mask) }.into())
    }

    fn is_null_sensitive(&self, _instance: &Self::Instance) -> bool {
        true
    }

    fn is_fallible(&self, _instance: &Self::Instance) -> bool {
        // If this type-checks its infallible.
        false
    }
}

/// Creates an expression that selects (includes) specific fields from an array.
///
/// Projects only the specified fields from the child expression, which must be of DType struct.
/// ```rust
/// # use vortex_array::expr::{select, root};
/// let expr = select(["name", "age"], root());
/// ```
pub fn select(field_names: impl Into<FieldNames>, child: Expression) -> Expression {
    Select
        .try_new_expr(FieldSelection::Include(field_names.into()), [child])
        .vortex_expect("Failed to create Select expression")
}

/// Creates an expression that excludes specific fields from an array.
///
/// Projects all fields except the specified ones from the input struct expression.
///
/// ```rust
/// # use vortex_array::expr::{select_exclude, root};
/// let expr = select_exclude(["internal_id", "metadata"], root());
/// ```
pub fn select_exclude(fields: impl Into<FieldNames>, child: Expression) -> Expression {
    Select
        .try_new_expr(FieldSelection::Exclude(fields.into()), [child])
        .vortex_expect("Failed to create Select expression")
}

impl ExpressionView<'_, Select> {
    pub fn child(&self) -> &Expression {
        &self.children()[0]
    }

    /// Turn the select expression into an `include`, relative to a provided array of field names.
    ///
    /// For example:
    /// ```rust
    /// # use vortex_array::expr::{root, Select};
    /// # use vortex_array::expr::{FieldSelection, select, select_exclude};
    /// # use vortex_dtype::FieldNames;
    /// let field_names = FieldNames::from(["a", "b", "c"]);
    /// let include = select(["a"], root());
    /// let exclude = select_exclude(["b", "c"], root());
    /// assert_eq!(
    ///     &include.as_::<Select>().as_include(&field_names).unwrap(),
    ///     &exclude.as_::<Select>().as_include(&field_names).unwrap(),
    /// );
    /// ```
    pub fn as_include(&self, field_names: &FieldNames) -> VortexResult<Expression> {
        Select.try_new_expr(
            FieldSelection::Include(self.data().as_include_names(field_names)?),
            [self.child().clone()],
        )
    }
}

impl FieldSelection {
    pub fn include(columns: FieldNames) -> Self {
        assert_eq!(columns.iter().unique().collect_vec().len(), columns.len());
        Self::Include(columns)
    }

    pub fn exclude(columns: FieldNames) -> Self {
        assert_eq!(columns.iter().unique().collect_vec().len(), columns.len());
        Self::Exclude(columns)
    }

    pub fn is_include(&self) -> bool {
        matches!(self, Self::Include(_))
    }

    pub fn is_exclude(&self) -> bool {
        matches!(self, Self::Exclude(_))
    }

    pub fn field_names(&self) -> &FieldNames {
        let (FieldSelection::Include(fields) | FieldSelection::Exclude(fields)) = self;

        fields
    }

    pub fn as_include_names(&self, field_names: &FieldNames) -> VortexResult<FieldNames> {
        if self
            .field_names()
            .iter()
            .any(|f| !field_names.iter().contains(f))
        {
            vortex_bail!(
                "Field {:?} in select not in field names {:?}",
                self,
                field_names
            );
        }
        match self {
            FieldSelection::Include(fields) => Ok(fields.clone()),
            FieldSelection::Exclude(exc_fields) => Ok(field_names
                .iter()
                .filter(|f| !exc_fields.iter().contains(f))
                .cloned()
                .collect()),
        }
    }
}

impl Display for FieldSelection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldSelection::Include(fields) => write!(f, "{{{}}}", DisplayFieldNames(fields)),
            FieldSelection::Exclude(fields) => write!(f, "~{{{}}}", DisplayFieldNames(fields)),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::FieldName;
    use vortex_dtype::FieldNames;
    use vortex_dtype::Nullability;

    use super::select;
    use super::select_exclude;
    use crate::arrays::StructArray;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::Select;
    use crate::expr::test_harness;
    use crate::IntoArray;
    use crate::ToCanonical;

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
    }

    #[test]
    pub fn include_columns() {
        let st = test_array();
        let select = select(vec![FieldName::from("a")], root());
        let selected = select.evaluate(&st.to_array()).unwrap().to_struct();
        let selected_names = selected.names().clone();
        assert_eq!(selected_names.as_ref(), &["a"]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = select_exclude(vec![FieldName::from("a")], root());
        let selected = select.evaluate(&st.to_array()).unwrap().to_struct();
        let selected_names = selected.names().clone();
        assert_eq!(selected_names.as_ref(), &["b"]);
    }

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();

        let select_expr = select(vec![FieldName::from("a")], root());
        let expected_dtype = DType::Struct(
            dtype
                .as_struct_fields_opt()
                .unwrap()
                .project(&["a".into()])
                .unwrap(),
            Nullability::NonNullable,
        );
        assert_eq!(select_expr.return_dtype(&dtype).unwrap(), expected_dtype);

        let select_expr_exclude = select_exclude(
            vec![
                FieldName::from("col1"),
                FieldName::from("col2"),
                FieldName::from("bool1"),
                FieldName::from("bool2"),
            ],
            root(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            expected_dtype
        );

        let select_expr_exclude = select_exclude(
            vec![FieldName::from("col1"), FieldName::from("col2")],
            root(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            DType::Struct(
                dtype
                    .as_struct_fields_opt()
                    .unwrap()
                    .project(&["a".into(), "bool1".into(), "bool2".into()])
                    .unwrap(),
                Nullability::NonNullable
            )
        );
    }

    #[test]
    fn test_as_include_names() {
        let field_names = FieldNames::from(["a", "b", "c"]);
        let include = select(["a"], root());
        let exclude = select_exclude(["b", "c"], root());
        assert_eq!(
            &include
                .as_::<Select>()
                .data()
                .as_include_names(&field_names)
                .unwrap(),
            &exclude
                .as_::<Select>()
                .data()
                .as_include_names(&field_names)
                .unwrap()
        );
    }
}
