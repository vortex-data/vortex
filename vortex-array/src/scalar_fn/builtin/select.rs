// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use prost::Message;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::proto::expr::FieldNames as ProtoFieldNames;
use crate::proto::expr::SelectOpts;
use crate::proto::expr::select_opts::Opts;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::builtin::field::DisplayFieldNames;
use crate::scalar_fn::builtin::pack::Pack;
use crate::scalar_fn::builtin::pack::PackOptions;
use crate::scalar_fn::display::ExprDisplay;

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
        static ID: CachedId = CachedId::new("vortex.select");
        *ID
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
            0 => ChildName::from("child"),
            _ => unreachable!(),
        }
    }

    fn fmt_sql(
        &self,
        selection: &FieldSelection,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        Display::fmt(expr.display_child(0), f)?;
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
        let child = args.get(0)?;
        if let Some(constant) = child.as_opt::<Constant>() {
            let child_struct_dtype = child
                .dtype()
                .as_struct_fields_opt()
                .ok_or_else(|| vortex_err!("Select child not a struct dtype"))?;
            let included = selection.normalize_to_included_fields(child_struct_dtype.names())?;
            let scalar = constant.scalar().as_struct().project(included.as_ref())?;

            return Ok(ConstantArray::new(scalar, child.len()).into_array());
        }

        let child = child.execute::<StructArray>(ctx)?;

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

    fn simplify<T: ReduceNode>(
        &self,
        selection: &FieldSelection,
        node: &T,
    ) -> VortexResult<Option<T>> {
        let child_struct = node.child(0);
        let struct_dtype = child_struct.node_dtype()?;
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

        // If no fields are included, we can trivially reduce to a pack node.
        // NOTE(ngates): we do this knowing that our layout expression partitioning logic has
        //  special-casing for pack, but not for select. We will fix this up when we revisit the
        //  layout APIs.
        if included_fields.is_empty() {
            return Ok(Some(node.new_node(
                Pack.bind(PackOptions {
                    names: FieldNames::empty(),
                    nullability: struct_nullability,
                }),
                &[],
            )?));
        }

        // We cannot always convert a `select` into a projection of its child fields. This is
        // because reading a field does a validity intersection of the struct validity with the
        // field, which is not the same as just "masking" out the unwanted fields (a selection).
        //
        // We can, however, make this reduction when the child of the `select` is already a
        // `pack` and we know that reading a field will do no validity intersections.
        //
        // The intersection only happens when the struct is nullable but the field is not. That
        // would change the semantics of a `select`, so we can only reduce when it won't happen.
        let would_intersect_validity =
            struct_nullability.is_nullable() && !all_included_fields_are_nullable;

        if let Some(child_pack) = child_struct.scalar_fn().and_then(|f| f.as_opt::<Pack>())
            && !would_intersect_validity
        {
            // Take the field nodes straight out of the child `pack` rather than wrap the
            // `pack` in `get_item`. The guard above proves the two are equivalent here, and the
            // optimizer visits children before their parent, so a `get_item` introduced now
            // would never be reduced away.
            let fields: Vec<T> = included_fields
                .iter()
                .map(|name| {
                    let idx = child_pack.names.find(name).ok_or_else(|| {
                        vortex_err!(
                            "Cannot find field {} in pack fields {:?}",
                            name,
                            child_pack.names
                        )
                    })?;

                    Ok(child_struct.child(idx))
                })
                .try_collect::<_, _, VortexError>()?;

            return Ok(Some(node.new_node(
                Pack.bind(PackOptions {
                    names: included_fields,
                    nullability: struct_nullability,
                }),
                &fields,
            )?));
        }

        Ok(None)
    }

    fn is_strict(&self, _options: &FieldSelection) -> bool {
        true
    }

    fn is_infallible(&self, _instance: &FieldSelection) -> bool {
        // If this type-checks, it is infallible.
        true
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
        // Hash the available fields so that wide selections stay linear.
        let available: HashSet<&FieldName> = available_fields.iter().collect();

        // Check that all of the field names exist in the available fields.
        if self.field_names().iter().any(|f| !available.contains(f)) {
            vortex_bail!(
                "Select fields {:?} must be a subset of child fields {:?}",
                self,
                available_fields
            );
        }

        match self {
            FieldSelection::Include(fields) => Ok(fields.clone()),
            FieldSelection::Exclude(exc_fields) => {
                let excluded: HashSet<&FieldName> = exc_fields.iter().collect();
                Ok(available_fields
                    .iter()
                    .filter(|f| !excluded.contains(f))
                    .cloned()
                    .collect())
            }
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
