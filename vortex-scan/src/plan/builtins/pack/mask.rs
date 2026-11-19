// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Moving expressions below the validity mask of a nullable struct's fields.

use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::NodeRewriter;
use vortex_array::expr::traversal::Transformed;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_error::VortexResult;

use crate::plan::builtins::zoned::holds_under_masking;

/// Moves field reads and statistics of `expr` below the masks over their input.
///
/// The partition rules read a field of a nullable struct as `mask(field, validity)`. An expression
/// over that mask stays above the struct, so this moves what it can below the mask, where the next
/// partitioning pushes it into the field's plan:
///
/// - `mask(s, v).name` becomes `mask(s.name, v)`: a field of a masked struct is null wherever the
///   struct is. Reading a nested field then reads only that field.
/// - `stat(mask(x, v), agg)` becomes `stat(x, agg)` when `agg` [holds under
///   masking](holds_under_masking), such as a bound, and every function above the statistic is
///   infallible. Masking only replaces values with nulls, so the statistic of `x` still holds for
///   `mask(x, v)`, if less tightly, and a pruning proof reads it only as such. The statistic of `x`
///   also covers the values that `v` hides, which a fallible function, such as a cast to a
///   narrower type, can fail on, although the rows never reach it.
/// - Any other statistic of a masked input becomes a null, which a pruning proof reads as a
///   statistic the zones do not store. A count over `x`, for one, misses the nulls `v` adds.
pub(super) fn lift_masks(expr: BoundExpression) -> VortexResult<BoundExpression> {
    Ok(expr.rewrite(&mut MaskLifter::default())?.into_inner())
}

#[derive(Default)]
struct MaskLifter {
    /// For each node on the path from the root to the current node, whether it is a fallible
    /// function.
    fallible: Vec<bool>,
}

impl NodeRewriter for MaskLifter {
    type NodeTy = BoundExpression;

    fn visit_down(&mut self, node: BoundExpression) -> VortexResult<Transformed<BoundExpression>> {
        let fallible = node
            .as_scalar()
            .is_some_and(|scalar_fn| !scalar_fn.signature().is_infallible());
        self.fallible.push(fallible);
        Ok(Transformed::no(node))
    }

    fn visit_up(&mut self, node: BoundExpression) -> VortexResult<Transformed<BoundExpression>> {
        self.fallible.pop();
        let Some(masked) = node.children().first().filter(|child| child.is::<Mask>()) else {
            return Ok(Transformed::no(node));
        };
        let (input, validity) = (masked.child(0).clone(), masked.child(1).clone());

        if let Some(name) = node.as_opt::<GetItem>() {
            let field = bound::get_item(name.clone(), input);
            return Ok(Transformed::yes(bound::mask(field, validity)));
        }

        let Some(options) = node.as_opt::<StatFn>() else {
            return Ok(Transformed::no(node));
        };
        if holds_under_masking(options.aggregate_fn()) && !self.fallible.contains(&true) {
            let unmasked = node.clone().with_children([input])?;
            // A statistic's dtype is nullable, so the unmasked input rarely changes it. Should it,
            // the expressions above it would change type, so the statistic is left unknown.
            if unmasked.dtype() == node.dtype() {
                return Ok(Transformed::yes(unmasked));
            }
        }
        let unknown = Scalar::null(node.dtype().clone());
        Ok(Transformed::yes(bound::lit(unknown)))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::nan_count::NanCount;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::cast;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::mask;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::fns::stat::StatFn;
    use vortex_array::scalar_fn::fns::stat::StatOptions;
    use vortex_error::VortexResult;

    use super::lift_masks;

    fn max() -> AggregateFnRef {
        Max.bind(NumericalAggregateOpts::skip_nans())
    }

    /// The statistic `aggregate_fn` of the field `f` of a nullable struct, read through the
    /// struct's validity.
    fn masked_stat(aggregate_fn: AggregateFnRef) -> Expression {
        StatFn.new_expr(
            StatOptions::new(aggregate_fn),
            [mask(get_item("f", root()), is_not_null(root()))],
        )
    }

    /// Lifts the masks of `expr` over a nullable `{f: f64}`, returning the result in display form.
    fn lifted(expr: Expression) -> VortexResult<String> {
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "f",
                DType::Primitive(PType::F64, Nullability::NonNullable),
            )]),
            Nullability::Nullable,
        );
        Ok(lift_masks(expr.bind(&dtype)?)?.to_string())
    }

    /// A statistic that holds under masking moves below the mask where only infallible functions
    /// read it. A fallible function above it, such as a cast to a narrower type, could fail on the
    /// values that the mask hides, and a count does not hold under masking, so both see a null.
    #[rstest]
    #[case::bound(
        lt_eq(masked_stat(max()), lit(5.0f64)),
        "(stat($.f, vortex.max()) <= 5f64)"
    )]
    #[case::no_nan(
        and(
            masked_stat(AllNonNan.bind(EmptyOptions)),
            lt_eq(masked_stat(max()), lit(5.0f64))
        ),
        "(stat($.f, vortex.all_non_nan()) and (stat($.f, vortex.max()) <= 5f64))"
    )]
    #[case::fallible(
        lt_eq(
            cast(masked_stat(max()), DType::Primitive(PType::I8, Nullability::Nullable)),
            lit(5i8)
        ),
        "(cast(null as i8?) <= 5i8)"
    )]
    #[case::count(eq(masked_stat(NanCount.bind(EmptyOptions)), lit(0u64)), "(null = 0u64)")]
    fn lifts_statistics_that_hold_where_they_are_read(
        #[case] expr: Expression,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        assert_eq!(lifted(expr)?, expected);
        Ok(())
    }
}
