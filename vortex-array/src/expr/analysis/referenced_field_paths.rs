// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::Field;
use crate::dtype::FieldPath;
use crate::dtype::FieldPathSet;
use crate::expr::Expression;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::root::Root;
use crate::scalar_fn::fns::select::Select;

/// Returns a prefix-minimal set of rooted field paths referenced by an expression.
///
/// A standalone root expression is represented by [`FieldPath::root`], which conservatively
/// selects all fields. When one referenced path is a prefix of another, only the prefix is returned.
/// Scalar functions other than `GetItem` and `Select` conservatively reference each complete child
/// output.
pub fn referenced_field_paths(expr: &Expression, scope: &DType) -> VortexResult<FieldPathSet> {
    // Validate the whole expression so plain GetItem paths and Select paths behave consistently.
    expr.return_dtype(scope)?;

    let mut field_paths = FieldPathSet::default();
    collect_referenced_field_paths(expr, scope, &FieldPath::root(), &mut field_paths)?;
    Ok(field_paths)
}

fn collect_referenced_field_paths(
    expr: &Expression,
    scope: &DType,
    requested_path: &FieldPath,
    field_paths: &mut FieldPathSet,
) -> VortexResult<()> {
    if expr.is::<Root>() {
        field_paths.insert_prefix(requested_path.clone());
        return Ok(());
    }

    if let Some(field_name) = expr.as_opt::<GetItem>() {
        let requested_path = FieldPath::from_iter(
            once(Field::Name(field_name.clone())).chain(requested_path.parts().iter().cloned()),
        );
        return collect_referenced_field_paths(expr.child(0), scope, &requested_path, field_paths);
    }

    if let Some(selection) = expr.as_opt::<Select>() {
        let child_dtype = expr.child(0).return_dtype(scope)?;
        let child_fields = child_dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Select child is not a struct"))?;
        let included_fields = selection.normalize_to_included_fields(child_fields.names())?;

        if requested_path.is_root() {
            for field_name in included_fields {
                collect_referenced_field_paths(
                    expr.child(0),
                    scope,
                    &FieldPath::from_name(field_name),
                    field_paths,
                )?;
            }
        } else if let Some(Field::Name(field_name)) = requested_path.parts().first()
            && included_fields
                .iter()
                .any(|included| included == field_name)
        {
            collect_referenced_field_paths(expr.child(0), scope, requested_path, field_paths)?;
        }
        return Ok(());
    }

    for child in expr.children().iter() {
        collect_referenced_field_paths(child, scope, &FieldPath::root(), field_paths)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::get_item;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::select;
    use crate::expr::select_exclude;

    fn scope() -> DType {
        DType::Struct(
            StructFields::from_iter([(
                "a",
                DType::Struct(
                    StructFields::from_iter([("x", I32), ("y", I32)]),
                    NonNullable,
                ),
            )]),
            NonNullable,
        )
    }

    #[test]
    fn nested_select_preserves_field_path() -> VortexResult<()> {
        let expr = select(["x"], get_item("a", root()));

        assert_eq!(
            referenced_field_paths(&expr, &scope())?,
            FieldPathSet::from_iter([FieldPath::from_name("a").push("x")])
        );
        Ok(())
    }

    #[test]
    fn get_item_after_select_only_references_requested_field() -> VortexResult<()> {
        let expr = get_item("x", select(["x", "y"], get_item("a", root())));

        assert_eq!(
            referenced_field_paths(&expr, &scope())?,
            FieldPathSet::from_iter([FieldPath::from_name("a").push("x")])
        );
        Ok(())
    }

    #[test]
    fn select_exclude_references_included_fields() -> VortexResult<()> {
        let expr = select_exclude(["y"], get_item("a", root()));

        assert_eq!(
            referenced_field_paths(&expr, &scope())?,
            FieldPathSet::from_iter([FieldPath::from_name("a").push("x")])
        );
        Ok(())
    }

    #[test]
    fn ancestor_path_subsumes_descendant() -> VortexResult<()> {
        let expr = pack(
            [
                ("a", get_item("a", root())),
                ("x", get_item("x", get_item("a", root()))),
            ],
            NonNullable,
        );

        assert_eq!(
            referenced_field_paths(&expr, &scope())?,
            FieldPathSet::from_iter([FieldPath::from_name("a")])
        );
        Ok(())
    }

    #[test]
    fn root_references_all_fields() -> VortexResult<()> {
        assert_eq!(
            referenced_field_paths(&root(), &scope())?,
            FieldPathSet::from_iter([FieldPath::root()])
        );
        Ok(())
    }

    #[test]
    fn invalid_get_item_path_returns_error() {
        assert!(referenced_field_paths(&get_item("missing", root()), &scope()).is_err());
    }
}
