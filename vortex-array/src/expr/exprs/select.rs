// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::FieldNames;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr::FieldNames as ProtoFieldNames;
use vortex_proto::expr::SelectOpts;
use vortex_proto::expr::select_opts::Opts;

use crate::IntoArray;
use crate::arrays::StructArray;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::ExprId;
use crate::expr::SimplifyCtx;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;
use crate::expr::field::DisplayFieldNames;
use crate::expr::get_item;
use crate::expr::pack;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldSelection {
    Include(FieldNames),
    Exclude(FieldNames),
}

pub struct Select;

impl VTable for Select {
    type Options = FieldSelection;

    fn id(&self) -> ExprId {
        ExprId::new_ref("vortex.select")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
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

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
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

        Ok(field_selection)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::new_ref("child"),
            _ => unreachable!(),
        }
    }

    fn fmt_sql(
        &self,
        selection: &FieldSelection,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        expr.child(0).fmt_sql(f)?;
        match selection {
            FieldSelection::Include(fields) => {
                write!(f, "{{{}}}", DisplayFieldNames(fields))
            }
            FieldSelection::Exclude(fields) => {
                write!(f, "{{~ {}}}", DisplayFieldNames(fields))
            }
        }
    }

    fn return_dtype(
        &self,
        selection: &FieldSelection,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let child_dtype = &arg_dtypes[0];
        let child_struct_dtype = child_dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Select child not a struct dtype"))?;

        let projected = match selection {
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

    fn execute(
        &self,
        selection: &FieldSelection,
        mut args: ExecutionArgs,
    ) -> VortexResult<ExecutionResult> {
        let child = args
            .inputs
            .pop()
            .vortex_expect("Missing input child")
            .execute::<StructArray>(args.ctx)?;

        let result = match selection {
            FieldSelection::Include(f) => child.project(f.as_ref()),
            FieldSelection::Exclude(names) => {
                let included_names = child
                    .names()
                    .iter()
                    .filter(|&f| !names.as_ref().contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                child.project(included_names.as_slice())
            }
        }?;

        result.into_array().execute(args.ctx)
    }

    fn simplify(
        &self,
        options: &Self::Options,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let child = expr.child(0);
        let child_dtype = ctx.return_dtype(child)?;
        let child_nullability = child_dtype.nullability();

        let child_dtype = child_dtype.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!(
                "Select child must return a struct dtype, however it was a {}",
                child_dtype
            )
        })?;

        let expr = pack(
            options
                .as_include_names(child_dtype.names())
                .map_err(|e| {
                    e.with_context(format!(
                        "Select fields {:?} must be a subset of child fields {:?}",
                        options,
                        child_dtype.names()
                    ))
                })?
                .iter()
                .map(|name| (name.clone(), get_item(name.clone(), child.clone()))),
            child_nullability,
        );

        Ok(Some(expr))
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _instance: &Self::Options) -> bool {
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
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::StructFields;

    use super::select;
    use super::select_exclude;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::StructArray;
    use crate::expr::exprs::pack::Pack;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::Select;
    use crate::expr::test_harness;

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
        let selected = st.to_array().apply(&select).unwrap().to_struct();
        let selected_names = selected.names().clone();
        assert_eq!(selected_names.as_ref(), &["a"]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = select_exclude(vec![FieldName::from("a")], root());
        let selected = st.to_array().apply(&select).unwrap().to_struct();
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
                .as_include_names(&field_names)
                .unwrap(),
            &exclude
                .as_::<Select>()
                .as_include_names(&field_names)
                .unwrap()
        );
    }

    #[test]
    fn test_remove_select_rule() {
        let dtype = DType::Struct(
            StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]),
            Nullable,
        );
        let e = select(["a", "b"], root());

        let result = e.optimize_recursive(&dtype).unwrap();

        assert!(result.is::<Pack>());
        assert!(result.return_dtype(&dtype).unwrap().is_nullable());
    }

    #[test]
    fn test_remove_select_rule_exclude_fields() {
        use crate::expr::exprs::select::select_exclude;

        let dtype = DType::Struct(
            StructFields::new(
                ["a", "b", "c"].into(),
                vec![I32.into(), I32.into(), I32.into()],
            ),
            Nullable,
        );
        let e = select_exclude(["c"], root());

        let result = e.optimize_recursive(&dtype).unwrap();

        assert!(result.is::<Pack>());

        // Should exclude "c" and include "a" and "b"
        let result_dtype = result.return_dtype(&dtype).unwrap();
        assert!(result_dtype.is_nullable());
        let fields = result_dtype.as_struct_fields_opt().unwrap();
        assert_eq!(fields.names().as_ref(), &["a", "b"]);
    }
}
