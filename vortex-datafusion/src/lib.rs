//! Connectors to enable [DataFusion](https://docs.rs/datafusion/latest/datafusion/) to read [`Vortex`](https://docs.rs/crate/vortex/latest) data.
#![deny(missing_docs)]
use std::fmt::Debug;

use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::stats::Precision as DFPrecision;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExprRef;
use datafusion::physical_plan::expressions::{BinaryExpr, Column, LikeExpr, Literal};
use vortex::stats::Precision;

mod convert;
mod persistent;

pub use persistent::*;

const SUPPORTED_BINARY_OPS: &[Operator] = &[
    Operator::Eq,
    Operator::NotEq,
    Operator::Gt,
    Operator::GtEq,
    Operator::Lt,
    Operator::LtEq,
];

fn supported_data_types(dt: DataType) -> bool {
    use DataType::*;
    let is_supported = dt.is_integer()
        || dt.is_floating()
        || dt.is_null()
        || matches!(
            dt,
            Boolean
                | Utf8
                | Utf8View
                | Binary
                | BinaryView
                | Date32
                | Date64
                | Timestamp(_, _)
                | Time32(_)
                | Time64(_)
        );

    if !is_supported {
        log::debug!("DataFusion data type {dt:?} is not supported");
    }

    is_supported
}

fn can_be_pushed_down(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    let expr = expr.as_any();
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        (binary.op().is_logic_operator() || SUPPORTED_BINARY_OPS.contains(binary.op()))
            && can_be_pushed_down(binary.left(), schema)
            && can_be_pushed_down(binary.right(), schema)
    } else if let Some(col) = expr.downcast_ref::<Column>() {
        schema
            .column_with_name(col.name())
            .map(|(_, field)| supported_data_types(field.data_type().clone()))
            .unwrap_or(false)
    } else if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        can_be_pushed_down(like.expr(), schema) && can_be_pushed_down(like.pattern(), schema)
    } else if let Some(lit) = expr.downcast_ref::<Literal>() {
        supported_data_types(lit.value().data_type())
    } else {
        log::debug!("DataFusion expression can't be pushed down: {expr:?}");
        false
    }
}

/// Extension trait to convert our [`Precision`](vortex::stats::Precision) to Datafusion's [`Precision`](datafusion_common::stats::Precision)
trait PrecisionExt<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    /// Convert `Precision` to the datafusion equivalent.
    fn to_df(self) -> DFPrecision<T>;
}

impl<T> PrecisionExt<T> for Precision<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Precision::Exact(v) => DFPrecision::Exact(v),
            Precision::Inexact(v) => DFPrecision::Inexact(v),
        }
    }
}

impl<T> PrecisionExt<T> for Option<Precision<T>>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Some(v) => v.to_df(),
            None => DFPrecision::Absent,
        }
    }
}
