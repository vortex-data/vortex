// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::lit;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_array::stats::bind::StatBinder;
use vortex_array::stats::bind::bind_stats;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::FileStatistics;

pub(crate) fn can_prune_file_stats(
    expr: &BoundExpression,
    row_count: u64,
    file_stats: &FileStatistics,
    struct_fields: &StructFields,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(pruning_expr) = expr.falsify(session)? else {
        return Ok(false);
    };

    let binder = FileStatsBinder {
        file_stats,
        struct_fields,
    };
    let pruning_expr = bind_stats(pruning_expr, &binder)?;

    if let Some(result) = pruning_expr.as_opt::<Literal>() {
        return Ok(result.as_bool().value() == Some(true));
    }

    let pruning = NullArray::new(1).into_array().apply_bound(&pruning_expr)?;
    let row_count_replacement = ConstantArray::new(row_count, pruning.len()).into_array();
    let pruning = substitute_row_count(pruning, &row_count_replacement)?;

    let mut ctx = session.create_execution_ctx();
    let result = pruning
        .execute::<Canonical>(&mut ctx)?
        .into_bool()
        .into_array()
        .execute_scalar(0, &mut ctx)?;

    Ok(result.as_bool().value() == Some(true))
}

struct FileStatsBinder<'a> {
    file_stats: &'a FileStatistics,
    struct_fields: &'a StructFields,
}

impl StatBinder for FileStatsBinder<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        // These aggregates have identical state and result semantics. Matching dtypes alone
        // does not establish that a finalized result can satisfy a reference to aggregate state.
        if !(aggregate_fn.is::<Min>()
            || aggregate_fn.is::<Max>()
            || aggregate_fn.is::<Sum>()
            || aggregate_fn.is::<NullCount>()
            || aggregate_fn.is::<NanCount>()
            || aggregate_fn.is::<UncompressedSizeInBytes>())
        {
            return Ok(None);
        }
        let Some(field_path) = direct_field_path(input) else {
            return Ok(None);
        };
        Ok(self.stat_ref(&field_path, aggregate_fn, stat_dtype))
    }
}

impl FileStatsBinder<'_> {
    fn stat_ref(
        &self,
        field_path: &FieldPath,
        aggregate: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> Option<BoundExpression> {
        // FileStats currently only holds top-level field statistics.
        if field_path.parts().len() != 1 {
            return None;
        }

        let field_name = field_path.parts()[0].as_name()?;
        let field_idx = self.struct_fields.find(field_name)?;
        let field_stats = self.file_stats.fields().get(field_idx)?;

        let value = field_stats.get(aggregate).as_exact()?;
        if !value.dtype().eq_ignore_nullability(stat_dtype) {
            return None;
        }
        Some(lit(value.cast(stat_dtype).ok()?))
    }
}

fn direct_field_path(expr: &BoundExpression) -> Option<FieldPath> {
    if expr.is_root() {
        return Some(FieldPath::root());
    }

    if expr.is::<Cast>() {
        return direct_field_path(expr.child(0));
    }

    let field_name = expr.as_opt::<GetItem>()?;
    direct_field_path(expr.child(0)).map(|path| path.push(field_name.clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::literal::Literal;
    use vortex_array::stats::AggregateResults;
    use vortex_array::stats::bind::bind_stats;
    use vortex_array::stats::stat;
    use vortex_error::VortexResult;

    use super::FileStatsBinder;
    use super::can_prune_file_stats;
    use crate::FileStatistics;

    #[rstest]
    #[case::above(false, "aaa999", 0)]
    #[case::below(true, "aaa123", 0)]
    #[case::matching(false, "aaa500", 1)]
    fn file_pruning_agrees_with_full_evaluation(
        #[case] below: bool,
        #[case] threshold: &'static str,
        #[case] matches: usize,
    ) -> VortexResult<()> {
        let session = array_session();
        let array = StructArray::from_fields(&[(
            "s",
            VarBinArray::from(vec!["aaa123", "aaa999"]).into_array(),
        )])?
        .into_array();
        let field = get_item("s", root());
        let expr = if below {
            lt(field, lit(threshold))
        } else {
            gt(field, lit(threshold))
        }
        .bind(array.dtype())?;
        let mut ctx = session.create_execution_ctx();
        let full = array.clone().apply_bound(&expr)?.execute::<BoolArray>(&mut ctx)?;
        assert_eq!(full.to_mask_fill_null_false(&mut ctx).true_count(), matches);

        let dtype = DType::Utf8(Nullability::NonNullable);
        let fields = array.dtype().as_struct_fields();
        for (min, max, can_prune) in [
            (Precision::Exact("aaa123"), Precision::Exact("aaa999"), matches == 0),
            (Precision::Inexact("aaa"), Precision::Inexact("aab"), false),
            (Precision::Absent, Precision::Absent, false),
        ] {
            let results = AggregateResults::try_new(
                &dtype,
                [
                    (
                        Min.bind(NumericalAggregateOpts::skip_nans()),
                        min.map(|s| Scalar::utf8(s, Nullability::Nullable)),
                    ),
                    (
                        Max.bind(NumericalAggregateOpts::skip_nans()),
                        max.map(|s| Scalar::utf8(s, Nullability::Nullable)),
                    ),
                ],
            )?;
            let stats = FileStatistics::new(Arc::from([results]), Arc::from([dtype.clone()]));
            assert_eq!(can_prune_file_stats(&expr, 2, &stats, fields, &session)?, can_prune);
        }
        Ok(())
    }

    #[rstest]
    #[case::constant(IsConstant.bind(EmptyOptions))]
    #[case::sorted(IsSorted.bind(IsSortedOptions { strict: false }))]
    fn finalized_flags_do_not_supply_partial_states(
        #[case] aggregate: AggregateFnRef,
    ) -> VortexResult<()> {
        let dtype = DType::from(PType::I32);
        let fields = StructFields::from_iter([("i", dtype.clone())]);
        let results = AggregateResults::try_new(
            &dtype,
            [(aggregate.clone(), Precision::Exact(true.into()))],
        )?;
        let stats = FileStatistics::new(Arc::from([results]), Arc::from([dtype]));
        let binder = FileStatsBinder {
            file_stats: &stats,
            struct_fields: &fields,
        };
        let expr = stat(get_item("i", root()), aggregate)
            .bind(&DType::Struct(fields.clone(), Nullability::NonNullable))?;
        let expected_dtype = expr.dtype().clone();
        let bound = bind_stats(expr, &binder)?;
        assert!(bound.as_::<Literal>().is_null());
        assert_eq!(bound.dtype(), &expected_dtype);
        Ok(())
    }
}
