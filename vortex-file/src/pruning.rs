// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::lit;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
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
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(pruning_expr) = expr.falsify(session)? else {
        return Ok(false);
    };

    let binder = FileStatsBinder { file_stats };
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
}

impl StatBinder for FileStatsBinder<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
            return Ok(None);
        };
        let Some(field_path) = direct_field_path(input) else {
            return Ok(None);
        };
        Ok(self.stat_ref(&field_path, stat))
    }
}

impl FileStatsBinder<'_> {
    fn stat_ref(&self, field_path: &FieldPath, stat: Stat) -> Option<BoundExpression> {
        let (field_stats, field_dtype) = self.file_stats.get_by_path(field_path)?;

        let stat_value = field_stats.get(stat).as_exact()?;
        let stat_dtype = stat.dtype(field_dtype)?;
        let stat_scalar = Scalar::try_new(stat_dtype, Some(stat_value)).ok()?;

        Some(lit(stat_scalar))
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
