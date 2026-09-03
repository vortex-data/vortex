// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use super::OptimizerRule;
use crate::dtype::FieldNames;
use crate::expr::BoundExpression;
use crate::expr::ExpressionId;
use crate::expr::bound;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::merge::DuplicateHandling;
use crate::scalar_fn::fns::merge::Merge;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::pack::PackOptions;
use crate::scalar_fn::fns::select::Select;

/// Replaces a field access on a pack with the corresponding packed expression.
///
/// # Example
///
/// ```text
/// original: get_item("b", pack([("a", a), ("b", b)], NonNullable))
/// rewritten: b
/// ```
#[derive(Debug)]
pub(crate) struct GetItemFromPack;

impl OptimizerRule for GetItemFromPack {
    fn expression_id(&self) -> ExpressionId {
        GetItem.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let field_name = expr.as_::<GetItem>();
        let child = expr.child(0);
        let Some(pack) = child.as_opt::<Pack>() else {
            return Ok(None);
        };
        let Some(idx) = pack.names.find(field_name) else {
            return Err(vortex_err!(
                "Cannot find field {field_name} in pack fields {:?}",
                pack.names
            ));
        };

        let mut field = child.child(idx).clone();
        if pack.nullability.is_nullable() {
            field = Mask.try_new_bound_expr(EmptyOptions, [field, bound::lit(true)])?;
        }
        Ok(Some(field))
    }
}

/// Lowers a merge of struct expressions into a pack while honoring its duplicate-field policy.
///
/// # Example
///
/// When `left` contains `a` and `right` contains `b`:
///
/// ```text
/// original: merge([left, right])
/// rewritten: pack([("a", get_item("a", left)), ("b", get_item("b", right))], NonNullable)
/// ```
#[derive(Debug)]
pub(crate) struct MergeToPack;

impl OptimizerRule for MergeToPack {
    fn expression_id(&self) -> ExpressionId {
        Merge.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let options = expr.as_::<Merge>();
        let mut names = Vec::with_capacity(expr.children().len() * 2);
        let mut sources = Vec::with_capacity(expr.children().len() * 2);
        let mut duplicate_names = HashSet::new();

        for child in expr.children() {
            let fields = child.dtype().as_struct_fields_opt().ok_or_else(|| {
                vortex_err!(
                    "Merge child must return a struct dtype, got {}",
                    child.dtype()
                )
            })?;
            for name in fields.names().iter() {
                if let Some(idx) = names.iter().position(|existing| existing == name) {
                    duplicate_names.insert(name.clone());
                    sources[idx] = child.clone();
                } else {
                    names.push(name.clone());
                    sources.push(child.clone());
                }
            }
        }

        if options == &DuplicateHandling::Error && !duplicate_names.is_empty() {
            vortex_bail!(
                "merge: duplicate fields in children: {}",
                duplicate_names.into_iter().format(", ")
            )
        }

        let children = names
            .iter()
            .zip(sources)
            .map(|(name, source)| GetItem.try_new_bound_expr(name.clone(), [source]))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Some(Pack.try_new_bound_expr(
            PackOptions {
                names: FieldNames::from(names),
                nullability: expr.dtype().nullability(),
            },
            children,
        )?))
    }
}

/// Lowers a selection from a pack into a smaller pack when struct validity is preserved.
///
/// # Example
///
/// ```text
/// original: select(["b"], pack([("a", a), ("b", b)], NonNullable))
/// rewritten: pack([("b", b)], NonNullable)
/// ```
#[derive(Debug)]
pub(crate) struct SelectFromPack;

impl OptimizerRule for SelectFromPack {
    fn expression_id(&self) -> ExpressionId {
        Select.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let selection = expr.as_::<Select>();
        let child = expr.child(0);
        let struct_dtype = child.dtype();
        let struct_nullability = struct_dtype.nullability();
        let struct_fields = struct_dtype.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!("Select child must return a struct dtype, however it was a {struct_dtype}")
        })?;
        let included_fields = selection.normalize_to_included_fields(struct_fields.names())?;

        // Pack expressions always have all-valid struct validity. Other struct expressions may
        // carry row validity that rebuilding them as a pack would discard, even with no fields.
        if !child.is::<Pack>() {
            return Ok(None);
        }

        if included_fields.is_empty() {
            return Ok(Some(Pack.try_new_bound_expr(
                PackOptions {
                    names: FieldNames::default(),
                    nullability: struct_nullability,
                },
                [],
            )?));
        }

        let all_included_fields_are_nullable = included_fields.iter().all(|name| {
            struct_fields
                .field(name)
                .vortex_expect("included select field must exist")
                .is_nullable()
        });
        let would_intersect_validity =
            struct_nullability.is_nullable() && !all_included_fields_are_nullable;
        if would_intersect_validity {
            return Ok(None);
        }

        let mut children = Vec::with_capacity(included_fields.len());
        for name in included_fields.iter() {
            children.push(GetItem.try_new_bound_expr(name.clone(), [child.clone()])?);
        }
        Ok(Some(Pack.try_new_bound_expr(
            PackOptions {
                names: included_fields,
                nullability: struct_nullability,
            },
            children,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::BoundExpression;
    use crate::expr::bound;
    use crate::expr::optimizer::BoundExpressionOptimizer;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::pack::Pack;

    fn optimize(expr: &BoundExpression) -> VortexResult<BoundExpression> {
        BoundExpressionOptimizer::default().optimize(expr)
    }

    #[test]
    fn get_item_from_nullable_pack_preserves_field_dtype() -> VortexResult<()> {
        let expr = bound::get_item(
            "a",
            bound::pack([("a", bound::lit(1i32))], Nullability::Nullable),
        );

        assert_eq!(
            optimize(&expr)?,
            bound::lit(Scalar::primitive(1i32, Nullability::Nullable))
        );
        Ok(())
    }

    #[test]
    fn merge_lowers_to_pack() -> VortexResult<()> {
        let expr = bound::merge([
            bound::pack([("a", bound::lit(1i32))], Nullability::NonNullable),
            bound::pack([("b", bound::lit(2i64))], Nullability::NonNullable),
        ]);
        let optimized = optimize(&expr)?;

        assert!(optimized.is::<Pack>());
        assert_eq!(
            optimized,
            bound::pack(
                [("a", bound::lit(1i32)), ("b", bound::lit(2i64))],
                Nullability::NonNullable
            )
        );
        Ok(())
    }

    #[test]
    fn select_from_pack_lowers_to_smaller_pack() -> VortexResult<()> {
        let expr = bound::select(
            ["b"],
            bound::pack(
                [("a", bound::lit(1i32)), ("b", bound::lit(2i64))],
                Nullability::NonNullable,
            ),
        );

        assert_eq!(
            optimize(&expr)?,
            bound::pack([("b", bound::lit(2i64))], Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn empty_select_from_root_is_not_rewritten() -> VortexResult<()> {
        let dtype = DType::struct_(
            [("a", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::Nullable,
        );
        let expr = bound::select(FieldNames::default(), bound::root(dtype));

        assert_eq!(optimize(&expr)?, expr);
        Ok(())
    }
}
