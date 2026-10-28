// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Nested projection for structs and lists of structs.
//!
//! Unlike `select`, this function can descend through a list element boundary. It is primarily
//! used by list layout readers to push a projection into the element reader while retaining the
//! original list offsets and validity.

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::str::from_utf8;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::StructArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::StructFields;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::display::ExprDisplay;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// One or more field-name paths, relative to a struct (or list element struct).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectPaths(
    /// Paths to preserve, with each path represented as individual field names.
    pub Vec<Vec<FieldName>>,
);

impl Display for ProjectPaths {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?}", self.0)
    }
}

/// Scalar function that projects nested struct fields within list elements.
#[derive(Clone)]
pub struct Project;

impl ScalarFnVTable for Project {
    type Options = ProjectPaths;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.project");
        *ID
    }

    fn serialize(&self, paths: &ProjectPaths) -> VortexResult<Option<Vec<u8>>> {
        let mut out = Vec::new();
        put_u32(&mut out, paths.0.len())?;
        for path in &paths.0 {
            put_u32(&mut out, path.len())?;
            for name in path {
                let bytes = name.as_ref().as_bytes();
                put_u32(&mut out, bytes.len())?;
                out.extend_from_slice(bytes);
            }
        }
        Ok(Some(out))
    }

    fn deserialize(&self, bytes: &[u8], _session: &VortexSession) -> VortexResult<ProjectPaths> {
        let mut cursor = 0;
        let n = take_u32(bytes, &mut cursor)?;
        if n > bytes.len().saturating_sub(cursor) / 4 {
            vortex_bail!("bad project path count");
        }
        let mut paths = Vec::with_capacity(n);
        for _ in 0..n {
            let m = take_u32(bytes, &mut cursor)?;
            if m > bytes.len().saturating_sub(cursor) / 4 {
                vortex_bail!("bad project field count");
            }
            let mut path = Vec::with_capacity(m);
            for _ in 0..m {
                let len = take_u32(bytes, &mut cursor)?;
                let end = cursor
                    .checked_add(len)
                    .ok_or_else(|| vortex_err!("bad project path"))?;
                let value = from_utf8(
                    bytes
                        .get(cursor..end)
                        .ok_or_else(|| vortex_err!("bad project path"))?,
                )
                .map_err(|e| vortex_err!("invalid utf8 in project path: {e}"))?;
                path.push(FieldName::from(value));
                cursor = end;
            }
            paths.push(path);
        }
        if cursor != bytes.len() {
            vortex_bail!("trailing bytes in project paths");
        }
        Ok(ProjectPaths(paths))
    }

    fn arity(&self, _: &ProjectPaths) -> Arity {
        Arity::Exact(1)
    }
    fn child_name(&self, _: &ProjectPaths, _: usize) -> ChildName {
        ChildName::from("child")
    }

    fn fmt_sql(
        &self,
        paths: &ProjectPaths,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> FmtResult {
        write!(f, "project({}, {})", expr.display_child(0), paths)
    }

    fn return_dtype(&self, paths: &ProjectPaths, args: &[DType]) -> VortexResult<DType> {
        project_dtype(&args[0], &paths.0)
    }

    fn execute(
        &self,
        paths: &ProjectPaths,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        project_array(args.get(0)?, &paths.0, ctx)
    }
}

fn put_u32(out: &mut Vec<u8>, value: usize) -> VortexResult<()> {
    out.extend_from_slice(
        &u32::try_from(value)
            .map_err(|_| vortex_err!("project path too large"))?
            .to_le_bytes(),
    );
    Ok(())
}
fn take_u32(bytes: &[u8], cursor: &mut usize) -> VortexResult<usize> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| vortex_err!("bad project path"))?;
    let value = u32::from_le_bytes(
        bytes
            .get(*cursor..end)
            .ok_or_else(|| vortex_err!("bad project path"))?
            .try_into()
            .map_err(|_| vortex_err!("bad project path"))?,
    );
    *cursor = end;
    Ok(value as usize)
}

fn project_dtype(dtype: &DType, paths: &[Vec<FieldName>]) -> VortexResult<DType> {
    for path in paths {
        validate_path(dtype, path)?;
    }
    match dtype {
        DType::List(element, nullability) => Ok(DType::List(
            Arc::new(project_dtype(element, paths)?),
            *nullability,
        )),
        DType::Struct(fields, nullability) => {
            for path in paths.iter().filter(|path| !path.is_empty()) {
                if fields.find(path[0].as_ref()).is_none() {
                    vortex_bail!("project path field {} does not exist in {dtype}", path[0]);
                }
            }
            let mut projected = Vec::new();
            for (name, field_dtype) in fields.names().iter().zip(fields.fields()) {
                let matching: Vec<_> = paths
                    .iter()
                    .filter(|path| path.first() == Some(name))
                    .collect();
                if matching.is_empty() {
                    continue;
                }
                let descendants: Vec<_> = matching
                    .iter()
                    .filter_map(|path| path.get(1..))
                    .filter(|p| !p.is_empty())
                    .map(|p| p.to_vec())
                    .collect();
                let has_whole_field = matching.iter().any(|p| p.len() == 1);
                let out_dtype = if has_whole_field || descendants.is_empty() {
                    field_dtype.clone()
                } else {
                    project_dtype(&field_dtype, &descendants)?
                };
                projected.push((name.clone(), out_dtype));
            }
            Ok(DType::Struct(
                StructFields::from_iter(projected),
                *nullability,
            ))
        }
        other => vortex_bail!("project paths require struct/list dtype, got {other}"),
    }
}

fn validate_path(dtype: &DType, path: &[FieldName]) -> VortexResult<()> {
    if path.is_empty() {
        return Ok(());
    }
    match dtype {
        DType::List(element, _) => validate_path(element, path),
        DType::Struct(fields, _) => {
            let name = &path[0];
            let field = fields.field(name).ok_or_else(|| {
                vortex_err!("project path field {name} does not exist in {dtype}")
            })?;
            validate_path(&field, &path[1..])
        }
        other => vortex_bail!("project path continues past non-struct dtype {other}"),
    }
}

fn project_array(
    input: ArrayRef,
    paths: &[Vec<FieldName>],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match input.dtype().clone() {
        DType::Struct(..) => project_struct(input.execute::<StructArray>(ctx)?, paths, ctx),
        DType::List(..) => {
            let list = input
                .execute::<Canonical>(ctx)?
                .into_array()
                .try_downcast::<ListView>()
                .map_err(|_| vortex_err!("canonical list projection failed"))?;
            let elements = project_array(list.elements().clone(), paths, ctx)?;
            Ok(ListViewArray::try_new(
                elements,
                list.offsets().clone(),
                list.sizes().clone(),
                list.listview_validity(),
                ctx,
            )?
            .into_array())
        }
        other => vortex_bail!("project requires struct or list-of-struct, got {other}"),
    }
}

fn project_struct(
    input: StructArray,
    paths: &[Vec<FieldName>],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut names = Vec::new();
    let mut columns = Vec::new();
    for name in input.names().iter() {
        let matching: Vec<_> = paths
            .iter()
            .filter(|path| path.first() == Some(name))
            .collect();
        if matching.is_empty() {
            continue;
        }
        names.push(name.clone());
        let descendants: Vec<_> = matching
            .iter()
            .filter_map(|path| path.get(1..))
            .filter(|p| !p.is_empty())
            .map(|p| p.to_vec())
            .collect();
        if matching.iter().any(|p| p.len() == 1) || descendants.is_empty() {
            columns.push(input.unmasked_field_by_name(name.as_ref())?.clone());
        } else {
            columns.push(project_array(
                input.unmasked_field_by_name(name.as_ref())?.clone(),
                &descendants,
                ctx,
            )?);
        }
    }
    let validity = input.struct_validity();
    Ok(StructArray::try_new(
        FieldNames::from(names.as_slice()),
        columns,
        input.len(),
        validity,
    )?
    .into_array())
}

/// Build a nested projection expression. Paths are relative to `dtype`; an empty path selects
/// the complete input. `dtype` is used to validate and type-check the expression before execution.
pub fn project_paths(
    paths: &[Vec<FieldName>],
    child: Expression,
    dtype: &DType,
) -> VortexResult<Expression> {
    if paths.iter().any(Vec::is_empty) {
        return Ok(child);
    }
    drop(project_dtype(dtype, paths)?);
    Ok(Project.new_expr(ProjectPaths(paths.to_vec()), vec![child]))
}

/// Bound-expression equivalent of [`project_paths`], for layout readers which have already
/// expanded a struct root into individual field expressions.
pub fn project_paths_bound(
    paths: &[Vec<FieldName>],
    child: BoundExpression,
) -> VortexResult<BoundExpression> {
    if paths.iter().any(Vec::is_empty) {
        return Ok(child);
    }
    Project.try_new_bound_expr(ProjectPaths(paths.to_vec()), vec![child])
}

/// Build a projection for the fields inside each list element.
pub fn list_project(
    paths: &[Vec<FieldName>],
    child: Expression,
    dtype: &DType,
) -> VortexResult<Expression> {
    match dtype {
        DType::List(..) => project_paths(paths, child, dtype),
        other => vortex_bail!("list_project requires a list dtype, got {other}"),
    }
}

/// Project a struct expression using an explicit list of nested paths.
pub fn struct_project(
    paths: &[Vec<FieldName>],
    child: Expression,
    dtype: &DType,
) -> VortexResult<Expression> {
    match dtype {
        DType::Struct(..) => project_paths(paths, child, dtype),
        other => vortex_bail!("struct_project requires a struct dtype, got {other}"),
    }
}
