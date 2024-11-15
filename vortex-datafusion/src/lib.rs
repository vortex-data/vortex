//! Connectors to enable DataFusion to read Vortex data.

#![allow(clippy::nonminimal_bool)]

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion_common::Result as DFResult;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::{Expr, Operator};
use persistent::config::VortexTableOptions;
use persistent::provider::VortexFileTableProvider;
use vortex_array::{ArrayDType, ArrayData};
use vortex_error::vortex_err;

use crate::memory::{VortexMemTable, VortexMemTableOptions};

pub mod memory;
pub mod persistent;

const SUPPORTED_BINARY_OPS: &[Operator] = &[
    Operator::Eq,
    Operator::NotEq,
    Operator::Gt,
    Operator::GtEq,
    Operator::Lt,
    Operator::LtEq,
];

fn supported_data_types(dt: DataType) -> bool {
    dt.is_integer()
        || dt.is_floating()
        || dt.is_null()
        || dt == DataType::Boolean
        || dt == DataType::Binary
        || dt == DataType::Utf8
        || dt == DataType::Binary
        || dt == DataType::BinaryView
        || dt == DataType::Utf8View
        || dt == DataType::Date32
        || dt == DataType::Date64
        || matches!(
            dt,
            DataType::Timestamp(_, _) | DataType::Time32(_) | DataType::Time64(_)
        )
}

pub trait SessionContextExt {
    fn register_mem_vortex<S: AsRef<str>>(&self, name: S, array: ArrayData) -> DFResult<()> {
        self.register_mem_vortex_opts(name, array, VortexMemTableOptions::default())
    }

    fn register_mem_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        array: ArrayData,
        options: VortexMemTableOptions,
    ) -> DFResult<()>;

    fn read_mem_vortex(&self, array: ArrayData) -> DFResult<DataFrame> {
        self.read_mem_vortex_opts(array, VortexMemTableOptions::default())
    }

    fn read_mem_vortex_opts(
        &self,
        array: ArrayData,
        options: VortexMemTableOptions,
    ) -> DFResult<DataFrame>;

    fn register_disk_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<()>;

    fn read_disk_vortex_opts(
        &self,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<DataFrame>;
}

impl SessionContextExt for SessionContext {
    fn register_mem_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        array: ArrayData,
        options: VortexMemTableOptions,
    ) -> DFResult<()> {
        if !array.dtype().is_struct() {
            return Err(vortex_err!(
                "Vortex arrays must have struct type, found {}",
                array.dtype()
            )
            .into());
        }

        let vortex_table = VortexMemTable::new(array, options);
        self.register_table(name.as_ref(), Arc::new(vortex_table))
            .map(|_| ())
    }

    fn read_mem_vortex_opts(
        &self,
        array: ArrayData,
        options: VortexMemTableOptions,
    ) -> DFResult<DataFrame> {
        if !array.dtype().is_struct() {
            return Err(vortex_err!(
                "Vortex arrays must have struct type, found {}",
                array.dtype()
            )
            .into());
        }

        let vortex_table = VortexMemTable::new(array, options);

        self.read_table(Arc::new(vortex_table))
    }

    fn register_disk_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<()> {
        let provider = Arc::new(VortexFileTableProvider::try_new(url, options)?);
        self.register_table(name.as_ref(), provider as _)?;

        Ok(())
    }

    fn read_disk_vortex_opts(
        &self,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<DataFrame> {
        let provider = Arc::new(VortexFileTableProvider::try_new(url, options)?);
        self.read_table(provider)
    }
}

fn can_be_pushed_down(expr: &Expr, schema: &Schema) -> bool {
    match expr {
        Expr::BinaryExpr(expr)
            if expr.op.is_logic_operator() || SUPPORTED_BINARY_OPS.contains(&expr.op) =>
        {
            can_be_pushed_down(expr.left.as_ref(), schema)
                & can_be_pushed_down(expr.right.as_ref(), schema)
        }
        Expr::Column(col) => match schema.column_with_name(col.name()) {
            Some((_, field)) => supported_data_types(field.data_type().clone()),
            _ => false,
        },
        Expr::Literal(lit) => supported_data_types(lit.data_type()),
        _ => false,
    }
}
