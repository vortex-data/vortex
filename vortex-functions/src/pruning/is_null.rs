// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Statistics-based pruning rewrites for [`IsNull`].

use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_expr::bound::eq;
use vortex_expr::bound::lit;
use vortex_expr::stats::rewrite::StatsRewriteRule;
use vortex_expr::stats::rewrite::helpers::*;
use vortex_session::VortexSession;

use crate::is_null::IsNull;

#[derive(Debug)]
pub(super) struct IsNullNullCountStatsRewrite;

impl StatsRewriteRule for IsNullNullCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNull.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0)).map(|null_count| eq(null_count, lit(0u64))))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0)).map(|null_count| eq(null_count, row_count())))
    }
}

#[derive(Debug)]
pub(super) struct IsNullAllNonNullStatsRewrite;

impl StatsRewriteRule for IsNullAllNonNullStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNull.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(Some(all_non_null(expr.child(0))))
    }
}

#[derive(Debug)]
pub(super) struct IsNullAllNullStatsRewrite;

impl StatsRewriteRule for IsNullAllNullStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNull.id()
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(Some(all_null(expr.child(0))))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::stats::Stat;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_error::VortexResult;
    use vortex_expr::Expression;
    use vortex_expr::ScalarFnExprExt;
    use vortex_expr::col;
    use vortex_expr::eq;
    use vortex_expr::lambda::row_count::RowCount;
    use vortex_expr::lit;
    use vortex_expr::or;
    use vortex_expr::stats::all_non_null;
    use vortex_expr::stats::all_null;
    use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
    use vortex_session::VortexSession;

    use crate::expr::is_null;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn scope() -> DType {
        DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(PType::I32, Nullability::Nullable)],
            ),
            Nullability::NonNullable,
        )
    }

    fn stat(expr: Expression, stat: Stat) -> Expression {
        let aggregate_fn = stat.aggregate_fn().expect("stat should have aggregate fn");
        vortex_expr::stats::stat(expr, aggregate_fn)
    }

    fn bind(expr: Expression) -> VortexResult<BoundExpression> {
        expr.bind(&scope())
    }

    #[test]
    fn falsifies_with_null_count_or_all_non_null() -> VortexResult<()> {
        let expected = or(
            eq(stat(col("a"), Stat::NullCount), lit(0u64)),
            all_non_null(col("a")),
        );
        assert_eq!(
            bind(is_null(col("a")))?.falsify(&SESSION)?,
            Some(bind(expected)?)
        );
        Ok(())
    }

    #[test]
    fn satisfies_with_null_count_or_all_null() -> VortexResult<()> {
        let expected = or(
            eq(
                stat(col("a"), Stat::NullCount),
                RowCount.new_expr(EmptyOptions, []),
            ),
            all_null(col("a")),
        );
        assert_eq!(
            bind(is_null(col("a")))?.satisfy(&SESSION)?,
            Some(bind(expected)?)
        );
        Ok(())
    }
}
