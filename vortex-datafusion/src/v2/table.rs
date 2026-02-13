// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VortexTable`] implements DataFusion's [`TableProvider`] trait, providing a direct
//! integration between a Vortex [`DataSource`] and DataFusion's query engine.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_catalog::TableProvider;
use datafusion_common::ColumnStatistics;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_expr::Expr;
use datafusion_expr::Operator as DFOperator;
use datafusion_expr::TableProviderFilterPushDown;
use datafusion_expr::TableType;
use datafusion_physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use vortex::compute::LikeOptions;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::arrow::FromArrowType;
use vortex::expr::Binary;
use vortex::expr::Expression;
use vortex::expr::Like;
use vortex::expr::Operator;
use vortex::expr::VTableExt;
use vortex::expr::and_collect;
use vortex::expr::cast;
use vortex::expr::get_item;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::scalar::Scalar;
use vortex::scan::api::DataSourceRef;
use vortex::scan::api::ScanRequest;
use vortex::session::VortexSession;

use crate::convert::FromDataFusion;
use crate::v2::exec::VortexExec;

fn vx_err(e: vortex::error::VortexError) -> datafusion_common::DataFusionError {
    datafusion_common::DataFusionError::External(Box::new(e))
}

/// Try to convert a DataFusion logical [`Expr`] into a Vortex [`Expression`].
///
/// Returns `None` if the expression contains unsupported nodes.
fn try_convert_expr(expr: &Expr) -> Option<Expression> {
    match expr {
        Expr::Column(col) => Some(get_item(col.name.clone(), root())),
        Expr::Literal(value, _) => Some(lit(Scalar::from_df(value))),
        Expr::BinaryExpr(binary) => {
            let left = try_convert_expr(&binary.left)?;
            let right = try_convert_expr(&binary.right)?;
            let op = try_convert_operator(&binary.op)?;
            Some(Binary.new_expr(op, [left, right]))
        }
        Expr::Not(child) => Some(not(try_convert_expr(child)?)),
        Expr::IsNull(child) => Some(is_null(try_convert_expr(child)?)),
        Expr::IsNotNull(child) => Some(not(is_null(try_convert_expr(child)?))),
        Expr::Like(like) => {
            let child = try_convert_expr(&like.expr)?;
            let pattern = try_convert_expr(&like.pattern)?;
            Some(Like.new_expr(
                LikeOptions {
                    negated: like.negated,
                    case_insensitive: like.case_insensitive,
                },
                [child, pattern],
            ))
        }
        Expr::Cast(cast_expr) => {
            let child = try_convert_expr(&cast_expr.expr)?;
            let target = DType::from_arrow((&cast_expr.data_type, Nullability::Nullable));
            Some(cast(child, target))
        }
        Expr::InList(in_list) => {
            let value = try_convert_expr(&in_list.expr)?;
            let scalars: Option<Vec<Scalar>> = in_list
                .list
                .iter()
                .map(|e| match e {
                    Expr::Literal(v, _) => Some(Scalar::from_df(v)),
                    _ => None,
                })
                .collect();
            let scalars = scalars?;
            let first_dtype = scalars.first()?.dtype().clone();
            let list_scalar = Scalar::list(first_dtype, scalars, Nullability::Nullable);
            let expr = list_contains(lit(list_scalar), value);
            if in_list.negated {
                Some(not(expr))
            } else {
                Some(expr)
            }
        }
        _ => None,
    }
}

fn try_convert_operator(op: &DFOperator) -> Option<Operator> {
    match op {
        DFOperator::Eq => Some(Operator::Eq),
        DFOperator::NotEq => Some(Operator::NotEq),
        DFOperator::Lt => Some(Operator::Lt),
        DFOperator::LtEq => Some(Operator::Lte),
        DFOperator::Gt => Some(Operator::Gt),
        DFOperator::GtEq => Some(Operator::Gte),
        DFOperator::And => Some(Operator::And),
        DFOperator::Or => Some(Operator::Or),
        DFOperator::Plus => Some(Operator::Add),
        DFOperator::Minus => Some(Operator::Sub),
        DFOperator::Multiply => Some(Operator::Mul),
        DFOperator::Divide => Some(Operator::Div),
        _ => None,
    }
}

/// A DataFusion [`TableProvider`] backed by a Vortex [`DataSourceRef`].
///
/// Maps each Vortex scan split to one DataFusion partition, letting DataFusion's scheduler
/// control concurrency.
pub struct VortexTable {
    data_source: DataSourceRef,
    session: VortexSession,
    arrow_schema: SchemaRef,
}

impl fmt::Debug for VortexTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexTable")
            .field("schema", &self.arrow_schema)
            .finish()
    }
}

impl VortexTable {
    /// Creates a new [`VortexTable`] from a Vortex data source and session.
    ///
    /// The Arrow schema will be used to emit the correct column names and types to DataFusion.
    /// The Vortex DType of the data source should be compatible with this Arrow schema.
    pub fn new(
        data_source: DataSourceRef,
        session: VortexSession,
        arrow_schema: SchemaRef,
    ) -> Self {
        Self {
            data_source,
            session,
            arrow_schema,
        }
    }
}

#[async_trait]
impl TableProvider for VortexTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.arrow_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|expr| {
                if try_convert_expr(expr).is_some() {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Build the projection expression and projected arrow schema.
        let (vx_projection, projected_schema) = if let Some(indices) = projection {
            let projected_fields: Vec<_> = indices
                .iter()
                .map(|&i| self.arrow_schema.field(i).clone())
                .collect();
            let projected_schema = Arc::new(arrow_schema::Schema::new(projected_fields));

            let elements: Vec<(String, Expression)> = indices
                .iter()
                .map(|&i| {
                    let name = self.arrow_schema.field(i).name().clone();
                    (name.clone(), get_item(name, root()))
                })
                .collect();
            let expr = pack(elements, Nullability::NonNullable);
            (Some(expr), projected_schema)
        } else {
            (None, Arc::clone(&self.arrow_schema))
        };

        // Convert logical filter expressions to Vortex expressions.
        let vx_filter = if !filters.is_empty() {
            let vx_exprs: Vec<Expression> = filters.iter().filter_map(try_convert_expr).collect();
            and_collect(vx_exprs)
        } else {
            None
        };

        // Build the scan request.
        let scan_request = ScanRequest {
            projection: vx_projection,
            filter: vx_filter,
            limit: limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
            ..Default::default()
        };

        // Create the scan and collect splits.
        let scan = self.data_source.scan(scan_request).map_err(vx_err)?;
        let splits: Vec<_> = scan.splits().try_collect().await.map_err(vx_err)?;

        Ok(Arc::new(VortexExec::new(
            splits,
            projected_schema,
            self.session.clone(),
        )))
    }

    fn statistics(&self) -> Option<Statistics> {
        let row_count_est = self.data_source.row_count_estimate();
        let num_rows = match row_count_est.upper {
            Some(upper) if row_count_est.lower == upper => usize::try_from(upper)
                .map(Precision::Exact)
                .unwrap_or_default(),
            _ => Precision::Absent,
        };

        let byte_size_est = self.data_source.byte_size_estimate();
        let total_byte_size = match byte_size_est.upper {
            Some(upper) if byte_size_est.lower == upper => usize::try_from(upper)
                .map(Precision::Exact)
                .unwrap_or_default(),
            _ => Precision::Absent,
        };

        let column_statistics =
            vec![ColumnStatistics::new_unknown(); self.arrow_schema.fields.len()];

        Some(Statistics {
            num_rows,
            total_byte_size,
            column_statistics,
        })
    }
}
