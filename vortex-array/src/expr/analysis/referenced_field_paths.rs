// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::Field;
use crate::dtype::FieldPath;
use crate::dtype::FieldPathSet;
use crate::expr::BoundExpr;
use crate::expr::traversal::FoldDownContext;
use crate::expr::traversal::FoldUp;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeFolderContext;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::select::Select;

/// Returns the rooted field paths referenced by an expression.
///
/// Iterating the returned set (via [`IntoIterator`]) yields the prefix-minimal covering set: when
/// one referenced path is a prefix of another, only the prefix is kept. A standalone root
/// expression is represented by [`FieldPath::root`], which conservatively selects all fields.
/// Scalar functions other than `GetItem` and `Select` conservatively reference each complete child
/// output.
// `scope` only feeds the debug-build invariant check below.
#[cfg_attr(not(debug_assertions), expect(unused_variables))]
pub fn referenced_field_paths(expr: &BoundExpr, scope: &DType) -> VortexResult<FieldPathSet> {
    let mut collector = ReferencedFieldPaths {
        field_paths: FieldPathSet::default(),
    };
    expr.clone()
        .fold_context(&vec![FieldPath::root()], &mut collector)?;
    let field_paths = collector.field_paths;

    // The top-level field of every referenced path must be one of the immediately accessed scope
    // fields: this analysis only refines *which nested fields* are read, never which top-level
    // fields. `FieldPath::root()` stands in for "all fields", so it expands to the whole scope.
    #[cfg(debug_assertions)]
    if let Some(scope_fields) = scope.as_struct_fields_opt() {
        use vortex_utils::aliases::hash_set::HashSet;

        use crate::dtype::FieldName;
        use crate::expr::analysis::immediate_access::immediate_scope_access;

        let referenced_heads: HashSet<FieldName> = if field_paths.iter().any(FieldPath::is_root) {
            scope_fields.names().iter().cloned().collect()
        } else {
            field_paths
                .iter()
                .filter_map(|path| match path.parts().first() {
                    Some(Field::Name(name)) => Some(name.clone()),
                    _ => None,
                })
                .collect()
        };
        debug_assert_eq!(
            referenced_heads,
            immediate_scope_access(expr, scope_fields),
            "referenced field path heads must match the immediately accessed scope fields"
        );
    }

    Ok(field_paths)
}

/// Threads the set of currently-requested field paths down the expression tree, narrowing it at
/// each `GetItem`/`Select`, and records the rooted paths reached at each `Root` leaf.
///
/// Paths are carried reversed so a `GetItem` can `push` its field instead of prepending it; they
/// are reversed back to rooted order when recorded at a `Root`, and `Select` reads a path's head
/// from its last element.
///
/// Narrowing is only sound through `GetItem` (a genuine field access) and `Select` (a genuine
/// column projection). Any other function is opaque—we cannot assume it preserves a field's
/// provenance—so its children conservatively re-request the whole scope, which is what keeps an
/// expression like `f($).x` reading every field of `$` rather than just `x`.
struct ReferencedFieldPaths {
    field_paths: FieldPathSet,
}

impl NodeFolderContext for ReferencedFieldPaths {
    type NodeTy = BoundExpr;
    type Result = ();
    type Context = Vec<FieldPath>;

    fn visit_down(
        &mut self,
        requested: &Self::Context,
        node: &BoundExpr,
    ) -> VortexResult<FoldDownContext<Self::Context, ()>> {
        if node.is_root() {
            self.field_paths.extend(
                requested
                    .iter()
                    .map(|path| FieldPath::from_iter(path.parts().iter().rev().cloned())),
            );
            return Ok(FoldDownContext::Skip(()));
        }

        if let Some(field_name) = node.as_opt::<GetItem>() {
            let appended = requested
                .iter()
                .map(|path| path.clone().push(Field::Name(field_name.clone())))
                .collect();
            return Ok(FoldDownContext::Continue(appended));
        }

        // Keep requested paths whose head is included, expanding a whole-scope request into one
        // path per included field.
        if let Some(selection) = node.as_opt::<Select>() {
            let child_dtype = node.child(0).dtype();
            let child_fields = child_dtype
                .as_struct_fields_opt()
                .ok_or_else(|| vortex_err!("Select child is not a struct"))?;
            let included_fields = selection.normalize_to_included_fields(child_fields.names())?;

            let mut narrowed = Vec::with_capacity(requested.len());
            for path in requested {
                if path.is_root() {
                    narrowed.extend(included_fields.iter().cloned().map(FieldPath::from_name));
                } else if let Some(Field::Name(field_name)) = path.parts().last()
                    && included_fields
                        .iter()
                        .any(|included| included == field_name)
                {
                    narrowed.push(path.clone());
                }
            }

            // Nothing is requested below this `Select`, so prune the subtree rather than letting an
            // opaque child re-request the whole scope.
            if narrowed.is_empty() {
                return Ok(FoldDownContext::Skip(()));
            }
            return Ok(FoldDownContext::Continue(narrowed));
        }

        // Any other function conservatively references each child's complete output.
        Ok(FoldDownContext::Continue(vec![FieldPath::root()]))
    }

    fn visit_up(
        &mut self,
        _node: BoundExpr,
        _requested: &Self::Context,
        _children: Vec<()>,
    ) -> VortexResult<FoldUp<()>> {
        Ok(FoldUp::Continue(()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::get_item;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::select;
    use crate::expr::select_exclude;
    use crate::expr::try_get_item;

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

    /// Collects the prefix-minimal field paths referenced by `expr` against [`scope`].
    fn referenced(expr: &BoundExpr) -> VortexResult<HashSet<FieldPath>> {
        Ok(referenced_field_paths(expr, &scope())?
            .into_iter()
            .collect())
    }

    #[test]
    fn nested_select_preserves_field_path() -> VortexResult<()> {
        let expr = select(["x"], get_item("a", root(scope())));

        assert_eq!(
            referenced(&expr)?,
            HashSet::from_iter([FieldPath::from_name("a").push("x")])
        );
        Ok(())
    }

    #[test]
    fn get_item_after_select_only_references_requested_field() -> VortexResult<()> {
        let expr = get_item("x", select(["x", "y"], get_item("a", root(scope()))));

        assert_eq!(
            referenced(&expr)?,
            HashSet::from_iter([FieldPath::from_name("a").push("x")])
        );
        Ok(())
    }

    #[test]
    fn select_exclude_references_included_fields() -> VortexResult<()> {
        let expr = select_exclude(["y"], get_item("a", root(scope())));

        assert_eq!(
            referenced(&expr)?,
            HashSet::from_iter([FieldPath::from_name("a").push("x")])
        );
        Ok(())
    }

    #[test]
    fn ancestor_path_subsumes_descendant() -> VortexResult<()> {
        let expr = pack(
            [
                ("a", get_item("a", root(scope()))),
                ("x", get_item("x", get_item("a", root(scope())))),
            ],
            NonNullable,
        );

        assert_eq!(
            referenced(&expr)?,
            HashSet::from_iter([FieldPath::from_name("a")])
        );
        Ok(())
    }

    #[test]
    fn get_item_through_opaque_fn_references_all_fields() -> VortexResult<()> {
        // `pack` is opaque to the path analysis: a `GetItem` of its output must not be pushed down
        // as a scope field access, so the wrapped `root()` conservatively references all fields.
        let expr = get_item("x", pack([("x", root(scope()))], NonNullable));

        assert_eq!(referenced(&expr)?, HashSet::from_iter([FieldPath::root()]));
        Ok(())
    }

    #[test]
    fn root_references_all_fields() -> VortexResult<()> {
        assert_eq!(
            referenced(&root(scope()))?,
            HashSet::from_iter([FieldPath::root()])
        );
        Ok(())
    }

    #[test]
    fn invalid_get_item_path_returns_error() {
        assert!(try_get_item("missing", root(scope())).is_err());
    }
}
