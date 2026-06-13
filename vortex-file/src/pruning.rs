// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpr;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_array::scalar_fn::internal::row_count::row_count;
use vortex_array::scalar_fn::internal::row_count::substitute_placeholders;
use vortex_array::stats::StatBinder;
use vortex_array::stats::bind_stats;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::FileStatistics;

/// Evaluates whether file-level footer stats prove `filter` cannot match this file.
pub(crate) fn evaluate_file_stats_pruning(
    filter: &BoundExpr,
    file_stats: &FileStatistics,
    file_dtype: &DType,
    row_count: u64,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(falsifier) = filter.falsify(session)? else {
        return Ok(false);
    };
    let Some(predicate) = bind_stats(
        falsifier,
        &mut FooterStatsBinder {
            file_stats,
            file_dtype,
        },
    )?
    else {
        return Ok(false);
    };

    if let Some(literal) = predicate.as_literal() {
        return Ok(literal.as_bool().value() == Some(true));
    }

    let pruning = NullArray::new(1).into_array().apply(&predicate)?;
    let row_count_replacement = ConstantArray::new(row_count, pruning.len()).into_array();
    let pruning = substitute_placeholders(pruning, &|placeholder| {
        placeholder
            .as_opt::<RowCount>()
            .map(|_| row_count_replacement.clone())
    })?;

    let mut ctx = session.create_execution_ctx();
    Ok(pruning.execute::<Mask>(&mut ctx)?.value(0))
}

struct FooterStatsBinder<'a> {
    file_stats: &'a FileStatistics,
    file_dtype: &'a DType,
}

impl StatBinder for FooterStatsBinder<'_> {
    fn bind_stat(
        &mut self,
        path: &FieldPath,
        stat: Stat,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpr>> {
        let Some((field_idx, field_dtype)) = self.field(path) else {
            return Ok(None);
        };
        let (stats, _) = self.file_stats.get(field_idx);
        let Some(stat_value) = stats.get(stat).as_exact() else {
            return Ok(None);
        };
        let Some(stat_dtype) = stat.dtype(field_dtype) else {
            return Ok(None);
        };
        Ok(Some(lit(Scalar::try_new(stat_dtype, Some(stat_value))?)))
    }

    fn bind_aggregate(
        &mut self,
        path: &FieldPath,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpr>> {
        let Some((_, input_dtype)) = self.field(path) else {
            return Ok(None);
        };

        if aggregate_fn.is::<AllNan>() {
            if !has_nans(input_dtype) {
                return Ok(Some(lit(false)));
            }
            return self.bind_count_check(path, Stat::NaNCount, row_count());
        }

        if aggregate_fn.is::<AllNonNan>() {
            if !has_nans(input_dtype) {
                return Ok(Some(lit(true)));
            }
            return self.bind_count_check(path, Stat::NaNCount, lit(0u64));
        }

        if aggregate_fn.is::<NanCount>() && !has_nans(input_dtype) {
            return Ok(Some(lit(0u64)));
        }

        if aggregate_fn.is::<AllNull>() {
            return self.bind_count_check(path, Stat::NullCount, row_count());
        }

        if aggregate_fn.is::<AllNonNull>() {
            return self.bind_count_check(path, Stat::NullCount, lit(0u64));
        }

        let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
            return Ok(None);
        };
        self.bind_stat(path, stat, stat_dtype)
    }
}

impl FooterStatsBinder<'_> {
    fn field(&self, path: &FieldPath) -> Option<(usize, &DType)> {
        // `.get()` rather than indexing: a caller may pair a file dtype with a shorter
        // FileStatistics (public constructors don't cross-validate); decline to prune
        // rather than panic.
        let Some(fields) = self.file_dtype.as_struct_fields_opt() else {
            return path
                .is_root()
                .then(|| self.file_stats.dtypes().first().map(|dtype| (0, dtype)))
                .flatten();
        };
        if path.parts().len() != 1 {
            return None;
        }
        let Field::Name(field_name) = &path.parts()[0] else {
            return None;
        };
        let field_idx = fields.find(field_name)?;
        let dtype = self.file_stats.dtypes().get(field_idx)?;
        Some((field_idx, dtype))
    }

    fn bind_count_check(
        &mut self,
        path: &FieldPath,
        stat: Stat,
        rhs: BoundExpr,
    ) -> VortexResult<Option<BoundExpr>> {
        let Some((_, input_dtype)) = self.field(path) else {
            return Ok(None);
        };
        let Some(dtype) = stat.dtype(input_dtype).map(|dtype| dtype.as_nullable()) else {
            return Ok(None);
        };
        let lhs = self
            .bind_stat(path, stat, &dtype)?
            .unwrap_or_else(|| lit(Scalar::null(dtype)));
        Ok(Some(eq(lhs, rhs)))
    }
}

fn has_nans(dtype: &DType) -> bool {
    matches!(dtype, DType::Primitive(ptype, _) if ptype.is_float())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::col;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::stats::StatsSession;
    use vortex_array::stats::StatsSet;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::evaluate_file_stats_pruning;
    use crate::FileStatistics;

    #[test]
    fn inexact_footer_stat_is_conservative() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "a",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let stats =
            StatsSet::from_iter([(Stat::Max, Precision::inexact(ScalarValue::from(10i32)))]);
        let file_stats = FileStatistics::new_with_dtype(Arc::from([stats]), &dtype);
        let expr = gt(col("a", &dtype), lit(10i32));

        assert!(!evaluate_file_stats_pruning(
            &expr,
            &file_stats,
            &dtype,
            1,
            &VortexSession::empty().with::<StatsSession>(),
        )?);
        Ok(())
    }

    #[test]
    fn computed_stat_input_does_not_bind_to_stored_field_max() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "a",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let stats = StatsSet::from_iter([(Stat::Max, Precision::exact(ScalarValue::from(10i32)))]);
        let file_stats = FileStatistics::new_with_dtype(Arc::from([stats]), &dtype);
        let expr = gt(checked_add(col("a", &dtype), lit(1i32)), lit(10i32));

        assert!(!evaluate_file_stats_pruning(
            &expr,
            &file_stats,
            &dtype,
            1,
            &VortexSession::empty().with::<StatsSession>(),
        )?);
        Ok(())
    }

    #[test]
    fn float_footer_pruning_uses_exact_zero_nan_count() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "f",
                DType::Primitive(PType::F32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let stats = StatsSet::from_iter([
            (Stat::Max, Precision::exact(ScalarValue::from(5.0f32))),
            (Stat::NaNCount, Precision::exact(ScalarValue::from(0u64))),
        ]);
        let file_stats = FileStatistics::new_with_dtype(Arc::from([stats]), &dtype);
        let expr = gt(col("f", &dtype), lit(10.0f32));

        assert!(evaluate_file_stats_pruning(
            &expr,
            &file_stats,
            &dtype,
            1,
            &VortexSession::empty().with::<StatsSession>(),
        )?);
        Ok(())
    }
}
