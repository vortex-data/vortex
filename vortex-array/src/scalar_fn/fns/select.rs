// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr::FieldNames as ProtoFieldNames;
use vortex_proto::expr::SelectOpts;
use vortex_proto::expr::select_opts::Opts;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::expr::expression::Expression;
use crate::expr::field::DisplayFieldNames;
use crate::expr::get_item;
use crate::expr::pack;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::pack::Pack;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldSelection {
    Include(FieldNames),
    Exclude(FieldNames),
}

#[derive(Clone)]
pub struct Select;

impl ScalarFnVTable for Select {
    type Options = FieldSelection;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.select")
    }

    fn serialize(&self, instance: &FieldSelection) -> VortexResult<Option<Vec<u8>>> {
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

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<FieldSelection> {
        let prost_metadata = SelectOpts::decode(_metadata)?;

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

    fn arity(&self, _options: &FieldSelection) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &FieldSelection, child_idx: usize) -> ChildName {
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
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?.execute::<StructArray>(ctx)?;

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

        result.into_array().execute(ctx)
    }

    fn simplify(
        &self,
        selection: &FieldSelection,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let child_struct = expr.child(0);
        let struct_dtype = ctx.return_dtype(child_struct)?;
        let struct_nullability = struct_dtype.nullability();

        let struct_fields = struct_dtype.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!(
                "Select child must return a struct dtype, however it was a {}",
                struct_dtype
            )
        })?;

        // "Mask" out the unwanted fields of the child struct `DType`.
        let included_fields = selection.normalize_to_included_fields(struct_fields.names())?;
        let all_included_fields_are_nullable = included_fields.iter().all(|name| {
            struct_fields
                .field(name)
                .vortex_expect(
                    "`normalize_to_included_fields` checks that the included fields already exist \
                     in `struct_fields`",
                )
                .is_nullable()
        });

        // If no fields are included, we can trivially simplify to a pack expression.
        // NOTE(ngates): we do this knowing that our layout expression partitioning logic has
        //  special-casing for pack, but not for select. We will fix this up when we revisit the
        //  layout APIs.
        if included_fields.is_empty() {
            let empty: Vec<(FieldName, Expression)> = vec![];
            return Ok(Some(pack(empty, struct_nullability)));
        }

        // We cannot always convert a `select` into a `pack(get_item(f1), get_item(f2), ...)`.
        // This is because `get_item` does a validity intersection of the struct validity with its
        // fields, which is not the same as just "masking" out the unwanted fields (a selection).
        //
        // We can, however, make this simplification when the child of the `select` is already a
        // `pack` and we know that `get_item` will do no validity intersections.
        let child_is_pack = child_struct.is::<Pack>();

        // `get_item` only performs validity intersection when the struct is nullable but the field
        // is not. This would change the semantics of a `select`, so we can only simplify when this
        // won't happen.
        let would_intersect_validity =
            struct_nullability.is_nullable() && !all_included_fields_are_nullable;

        if child_is_pack && !would_intersect_validity {
            let pack_expr = pack(
                included_fields
                    .into_iter()
                    .map(|name| (name.clone(), get_item(name, child_struct.clone()))),
                struct_nullability,
            );

            return Ok(Some(pack_expr));
        }

        Ok(None)
    }

    fn is_null_sensitive(&self, _instance: &FieldSelection) -> bool {
        true
    }

    fn is_fallible(&self, _instance: &FieldSelection) -> bool {
        // If this type-checks its infallible.
        false
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

    pub fn normalize_to_included_fields(
        &self,
        available_fields: &FieldNames,
    ) -> VortexResult<FieldNames> {
        // Check that all of the field names exist in the available fields.
        if self
            .field_names()
            .iter()
            .any(|f| !available_fields.iter().contains(f))
        {
            vortex_bail!(
                "Select fields {:?} must be a subset of child fields {:?}",
                self,
                available_fields
            );
        }

        match self {
            FieldSelection::Include(fields) => Ok(fields.clone()),
            FieldSelection::Exclude(exc_fields) => Ok(available_fields
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

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::struct_::StructArrayExt;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::root;
    use crate::expr::select;
    use crate::expr::select_exclude;
    use crate::expr::test_harness;
    use crate::scalar_fn::fns::select::Select;
    use crate::scalar_fn::fns::select::StructArray;

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
        let selected = st.into_array().apply(&select).unwrap().to_struct();
        let selected_names = selected.names().clone();
        assert_eq!(selected_names.as_ref(), &["a"]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = select_exclude(vec![FieldName::from("a")], root());
        let selected = st.into_array().apply(&select).unwrap().to_struct();
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
                .normalize_to_included_fields(&field_names)
                .unwrap(),
            &exclude
                .as_::<Select>()
                .normalize_to_included_fields(&field_names)
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

        assert!(result.return_dtype(&dtype).unwrap().is_nullable());
    }

    #[test]
    fn test_remove_select_rule_exclude_fields() {
        use crate::expr::select_exclude;

        let dtype = DType::Struct(
            StructFields::new(
                ["a", "b", "c"].into(),
                vec![I32.into(), I32.into(), I32.into()],
            ),
            Nullable,
        );
        let e = select_exclude(["c"], root());

        let result = e.optimize_recursive(&dtype).unwrap();

        // Should exclude "c" and include "a" and "b"
        let result_dtype = result.return_dtype(&dtype).unwrap();
        assert!(result_dtype.is_nullable());
        let fields = result_dtype.as_struct_fields_opt().unwrap();
        assert_eq!(fields.names().as_ref(), &["a", "b"]);
    }
}
