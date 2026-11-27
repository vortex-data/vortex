// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Narrows the struct columns of a raw scan to the children that DataFusion reads.
//!
//! When a projection stays in DataFusion, the scan reads each column that the projection references
//! as a raw input. A struct column that the projection reads only through [`get_field`] needs only
//! the children on those field paths. [`column_uses`] finds those paths, and [`prune_read`] narrows
//! the scan column to them. The column keeps its name and the DataFusion expressions do not change:
//! `get_field` finds a child by name and applies the nulls of every parent on the path, so it reads
//! the same values from the narrowed struct.
//!
//! A path ends at the result of one `get_field` call, and the scan reads that result whole.
//! DataFusion fixes the Arrow type of each call's result at planning time, so a call that returns a
//! struct must return all of it. In nested calls such as `get_field(get_field(s, "a"), "b")`, the
//! inner call returns `s.a`, so the scan reads all of `s.a`. DataFusion's simplifier flattens
//! nested calls into one call, `get_field(s, "a", "b")`, which reads only `s.a.b`.
//!
//! Any other use of a column, such as an argument to a function or a cast, reads it whole. A path
//! also stops narrowing where it cannot name one child for certain: at a key that is not a non-null
//! string literal, a missing child, a struct with duplicate child names, an Arrow extension type,
//! or a type that is not a struct.
//!
//! [`get_field`]: GetFieldFunc

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::select;

/// How a set of expressions uses a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FieldUse {
    /// The expressions read the whole field.
    Whole,
    /// The expressions read only these children of a struct field, keyed by child index.
    ///
    /// Only a field that [`struct_children`] accepts has child uses. The map is never empty, and it
    /// never holds a whole use for every child.
    Children(BTreeMap<usize, FieldUse>),
}

impl FieldUse {
    /// The use of `field` by a `get_field` call that reads `path` from it.
    fn of_path(field: &Field, path: &[&str]) -> Self {
        let Some((name, rest)) = path.split_first() else {
            return Self::Whole;
        };
        let Some(children) = struct_children(field) else {
            return Self::Whole;
        };
        let Some((child_idx, child)) = children.find(name) else {
            return Self::Whole;
        };

        Self::of_children(
            children,
            BTreeMap::from([(child_idx, Self::of_path(child, rest))]),
        )
    }

    /// The use of `field` by the expressions that use it as `self` and as `other`.
    ///
    /// A field that either side reads whole is read whole.
    fn merge(self, other: Self, field: &Field) -> Self {
        let (Self::Children(mut child_uses), Self::Children(other_child_uses), Some(children)) =
            (self, other, struct_children(field))
        else {
            return Self::Whole;
        };

        for (child_idx, other_use) in other_child_uses {
            let child_use = match child_uses.remove(&child_idx) {
                Some(child_use) => child_use.merge(other_use, &children[child_idx]),
                None => other_use,
            };
            child_uses.insert(child_idx, child_use);
        }

        Self::of_children(children, child_uses)
    }

    /// The use of a struct with `children` that reads `child_uses`, which is the whole struct when
    /// they read every child whole.
    fn of_children(children: &Fields, child_uses: BTreeMap<usize, FieldUse>) -> Self {
        let reads_every_child = child_uses.len() == children.len()
            && child_uses
                .values()
                .all(|child_use| *child_use == Self::Whole);
        if reads_every_child {
            return Self::Whole;
        }

        Self::Children(child_uses)
    }
}

/// Collect how `exprs` use each column of `schema` that they reference, keyed by column index.
///
/// A `get_field` read of a column uses only the children on its path. Every other reference reads
/// the column whole. Every key is a column index of `schema`, and a reference to a column outside
/// `schema` is an error.
pub(super) fn column_uses<'a>(
    exprs: impl IntoIterator<Item = &'a Arc<dyn PhysicalExpr>>,
    schema: &Schema,
) -> DFResult<BTreeMap<usize, FieldUse>> {
    let mut uses = BTreeMap::<usize, FieldUse>::new();
    let mut record = |column: &df_expr::Column, path: &[&str]| -> DFResult<()> {
        let field = schema.fields().get(column.index()).ok_or_else(|| {
            exec_datafusion_err!(
                "Column {}@{} is out of bounds",
                column.name(),
                column.index()
            )
        })?;

        let path_use = FieldUse::of_path(field, path);
        let column_use = match uses.remove(&column.index()) {
            Some(column_use) => column_use.merge(path_use, field),
            None => path_use,
        };
        uses.insert(column.index(), column_use);

        Ok(())
    };

    for expr in exprs {
        expr.apply(|node| {
            // The path already records its column. Skip the arguments of the call, because a visit
            // to the column below it would record a bare use, which reads the column whole.
            if let Some((column, path)) = field_path(node) {
                record(column, &path)?;
                return Ok(TreeNodeRecursion::Jump);
            }

            if let Some(column) = node.downcast_ref::<df_expr::Column>() {
                record(column, &[])?;
            }

            Ok(TreeNodeRecursion::Continue)
        })?;
    }

    Ok(uses)
}

/// Narrow `read`, the Vortex read of `field`, to the children that `field_use` keeps.
///
/// Returns the narrowed read and its Arrow field. The field keeps its name, nullability, and
/// metadata, and its kept children keep their Arrow fields, since they choose the Arrow types of
/// the scan output.
///
/// A struct whose kept children are all read whole becomes a [`select`], which keeps the struct's
/// validity. Otherwise the struct becomes a non-nullable [`pack`] of its narrowed children. Each
/// child is read through the struct, so it carries the struct's nulls, and `get_field` reads the
/// same values from either form. A pack that kept the struct's validity would have to know whether
/// the stored struct is nullable, which `field` does not tell: a source can declare a nullable
/// field over non-nullable storage.
pub(super) fn prune_read(
    read: Expression,
    field: &FieldRef,
    field_use: &FieldUse,
) -> (Expression, FieldRef) {
    let (FieldUse::Children(child_uses), Some(children)) = (field_use, struct_children(field))
    else {
        return (read, Arc::clone(field));
    };

    let (child_reads, child_fields): (Vec<_>, Vec<_>) = child_uses
        .iter()
        .map(|(&child_idx, child_use)| {
            let child = &children[child_idx];
            let child_read = get_item(child.name().as_str(), read.clone());
            let (child_read, child_field) = prune_read(child_read, child, child_use);
            ((child.name().clone(), child_read), child_field)
        })
        .unzip();

    let reads_whole_children = child_uses
        .values()
        .all(|child_use| *child_use == FieldUse::Whole);
    let pruned_read = if reads_whole_children {
        let names = child_reads
            .into_iter()
            .map(|(name, _)| name)
            .collect::<FieldNames>();
        select(names, read)
    } else {
        pack(child_reads, Nullability::NonNullable)
    };

    let pruned_field = field
        .as_ref()
        .clone()
        .with_data_type(DataType::Struct(child_fields.into()));

    (pruned_read, Arc::new(pruned_field))
}

/// The column and child names that `expr` reads, if `expr` is a `get_field` call on a column with
/// non-null string literal keys.
fn field_path(expr: &Arc<dyn PhysicalExpr>) -> Option<(&df_expr::Column, Vec<&str>)> {
    let get_field = ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(expr.as_ref())?;
    let [source, keys @ ..] = get_field.args() else {
        return None;
    };
    let column = source.downcast_ref::<df_expr::Column>()?;
    let keys = keys
        .iter()
        .map(|key| {
            key.downcast_ref::<df_expr::Literal>()?
                .value()
                .try_as_str()
                .flatten()
        })
        .collect::<Option<Vec<_>>>()?;

    Some((column, keys))
}

/// The children of `field`, if the scan can read them one at a time.
///
/// An Arrow extension type keeps the Vortex extension opaque, so reading its storage children
/// needs [`ext_storage`], and its plugin may export the storage differently. Duplicate child names
/// make a name-based read ambiguous.
///
/// [`ext_storage`]: vortex::expr::ext_storage
fn struct_children(field: &Field) -> Option<&Fields> {
    if field.metadata().contains_key(EXTENSION_TYPE_NAME_KEY) {
        return None;
    }
    let DataType::Struct(children) = field.data_type() else {
        return None;
    };

    children
        .iter()
        .map(|child| child.name())
        .all_unique()
        .then_some(children)
}

#[cfg(test)]
mod tests;
