// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Nested TPC-H tables, in the shape of Apache DataFusion Comet's [`benchmarks/nested`][comet].
//!
//! Join keys (the `*key` columns) and one partition column per table (`p_brand`,
//! `c_mktsegment`, `o_orderdate`, `l_shipdate`) stay at the top level. Every other column
//! moves into a one-element `<table>_data: list<struct<...>>`, so unnesting it never changes the
//! row count and the nested queries return the flat answers.
//!
//! [comet]: https://github.com/apache/datafusion-comet/tree/main/benchmarks/nested

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::StructArray;
use arrow_buffer::OffsetBuffer;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;

/// The partition columns of Comet's nested layout, kept at the top level.
const PARTITION_COLUMNS: &[&str] = &["p_brand", "c_mktsegment", "o_orderdate", "l_shipdate"];

/// Field name of each `<table>_data` list's element.
const ITEM: &str = "item";

/// Whether a flat column is a join key.
fn is_key(field: &Field) -> bool {
    field.name().ends_with("key")
}

/// Whether a flat column is a partition column.
fn is_partition(field: &Field) -> bool {
    PARTITION_COLUMNS.contains(&field.name().as_str())
}

/// The name of the list column holding the non-key columns of `table`.
pub fn data_column(table: &str) -> String {
    format!("{table}_data")
}

/// The nested schema of `table`, derived from its `flat` schema: the keys, then the partition
/// column, then `<table>_data`, each group in flat order.
pub fn nested_schema(table: &str, flat: &Schema) -> Schema {
    let keys = flat.fields().iter().filter(|field| is_key(field));
    let partition = flat.fields().iter().filter(|field| is_partition(field));
    let nested: Fields = flat
        .fields()
        .iter()
        .filter(|field| !is_key(field) && !is_partition(field))
        .cloned()
        .collect();
    let data = Field::new(
        data_column(table),
        DataType::List(Arc::new(Field::new(ITEM, DataType::Struct(nested), false))),
        false,
    );
    Schema::new(
        keys.chain(partition)
            .cloned()
            .chain(std::iter::once(Arc::new(data)))
            .collect::<Vec<FieldRef>>(),
    )
}

/// Nests one flat `batch` into `schema`, the result of [`nested_schema`] for its table.
pub fn nest_batch(schema: &SchemaRef, batch: &RecordBatch) -> Result<RecordBatch, ArrowError> {
    let column = |name: &str| -> Result<ArrayRef, ArrowError> {
        Ok(Arc::clone(batch.column(batch.schema().index_of(name)?)))
    };
    let columns = schema
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            DataType::List(item) => {
                let DataType::Struct(fields) = item.data_type() else {
                    return Err(ArrowError::SchemaError(format!(
                        "nested column {} must hold structs",
                        field.name()
                    )));
                };
                let children = fields
                    .iter()
                    .map(|child| column(child.name()))
                    .collect::<Result<Vec<_>, _>>()?;
                let values = StructArray::try_new(fields.clone(), children, None)?;
                let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(1, batch.num_rows()));
                let list = ListArray::try_new(Arc::clone(item), offsets, Arc::new(values), None)?;
                Ok(Arc::new(list) as ArrayRef)
            }
            _ => column(field.name()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// A reader of nested batches over a reader of flat batches of one table.
pub struct NestedBatchReader {
    inner: Box<dyn RecordBatchReader + Send>,
    schema: SchemaRef,
}

impl NestedBatchReader {
    pub fn new(table: &str, inner: Box<dyn RecordBatchReader + Send>) -> Self {
        let schema = Arc::new(nested_schema(table, &inner.schema()));
        Self { inner, schema }
    }
}

impl Iterator for NestedBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        Some(batch.and_then(|batch| nest_batch(&self.schema, &batch)))
    }
}

impl RecordBatchReader for NestedBatchReader {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::RecordBatchIterator;
    use arrow_array::cast::AsArray;
    use tpchgen::generators::LineItemGenerator;
    use tpchgen::generators::OrderGenerator;
    use tpchgen_arrow::LineItemArrow;
    use tpchgen_arrow::OrderArrow;

    use super::*;
    use crate::tpch::schema::CUSTOMER;
    use crate::tpch::schema::LINEITEM;
    use crate::tpch::schema::ORDERS;
    use crate::tpch::schema::REGION;

    fn names(fields: &Fields) -> Vec<&str> {
        fields.iter().map(|field| field.name().as_str()).collect()
    }

    fn data_fields<'a>(schema: &'a Schema, table: &str) -> &'a Fields {
        let DataType::List(item) = schema
            .field_with_name(&data_column(table))
            .unwrap()
            .data_type()
        else {
            panic!("{table}_data must be a list");
        };
        let DataType::Struct(fields) = item.data_type() else {
            panic!("{table}_data items must be structs");
        };
        fields
    }

    #[test]
    fn nested_schemas_follow_comet_layout() {
        let lineitem = nested_schema("lineitem", &LINEITEM);
        assert_eq!(
            names(lineitem.fields()),
            [
                "l_orderkey",
                "l_partkey",
                "l_suppkey",
                "l_shipdate",
                "lineitem_data"
            ]
        );
        assert_eq!(
            names(data_fields(&lineitem, "lineitem")),
            [
                "l_linenumber",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
                "l_returnflag",
                "l_linestatus",
                "l_commitdate",
                "l_receiptdate",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ]
        );

        let orders = nested_schema("orders", &ORDERS);
        assert_eq!(
            names(orders.fields()),
            ["o_orderkey", "o_custkey", "o_orderdate", "orders_data"]
        );
        let customer = nested_schema("customer", &CUSTOMER);
        assert_eq!(
            names(customer.fields()),
            ["c_custkey", "c_nationkey", "c_mktsegment", "customer_data"]
        );
        let region = nested_schema("region", &REGION);
        assert_eq!(names(region.fields()), ["r_regionkey", "region_data"]);
        assert_eq!(
            names(data_fields(&region, "region")),
            ["r_name", "r_comment"]
        );
    }

    #[test]
    fn nested_schema_keeps_flat_types() {
        let lineitem = nested_schema("lineitem", &LINEITEM);
        for field in data_fields(&lineitem, "lineitem").iter() {
            assert_eq!(
                field.as_ref(),
                LINEITEM.field_with_name(field.name()).unwrap(),
                "{} changed",
                field.name()
            );
        }
        assert!(
            lineitem.fields().iter().all(|field| !field.is_nullable()),
            "nested columns inherit non-nullability"
        );
    }

    #[test]
    fn nested_batches_hold_one_element_per_row() -> anyhow::Result<()> {
        let flat = OrderArrow::new(OrderGenerator::new(0.001, 1, 1)).with_batch_size(100);
        let flat_schema = flat.schema();
        let flat_batches: Vec<RecordBatch> = flat.collect::<Result<_, _>>()?;
        let reader = NestedBatchReader::new(
            "orders",
            Box::new(RecordBatchIterator::new(
                flat_batches.clone().into_iter().map(Ok),
                Arc::clone(&flat_schema),
            )),
        );
        assert_eq!(*reader.schema(), nested_schema("orders", &flat_schema));

        for (nested, flat) in reader.zip(flat_batches) {
            let nested = nested?;
            assert_eq!(nested.num_rows(), flat.num_rows());
            assert_eq!(
                nested.column_by_name("o_orderdate").unwrap(),
                flat.column_by_name("o_orderdate").unwrap()
            );
            let data = nested
                .column_by_name("orders_data")
                .unwrap()
                .as_list::<i32>();
            assert_eq!(data.null_count(), 0);
            assert!(
                data.offsets().windows(2).all(|pair| pair[1] - pair[0] == 1),
                "every list holds exactly one element"
            );
            let values = data.values().as_struct();
            assert_eq!(
                values.column_by_name("o_comment").unwrap(),
                flat.column_by_name("o_comment").unwrap()
            );
        }
        Ok(())
    }

    #[test]
    fn nested_lineitem_matches_generator_rows() -> anyhow::Result<()> {
        let flat = LineItemArrow::new(LineItemGenerator::new(0.001, 1, 1)).with_batch_size(1000);
        let rows: usize = flat
            .map(|batch| batch.map(|b| b.num_rows()))
            .sum::<Result<_, _>>()?;
        let flat = LineItemArrow::new(LineItemGenerator::new(0.001, 1, 1)).with_batch_size(1000);
        let nested = NestedBatchReader::new("lineitem", Box::new(flat));
        let nested_rows: usize = nested
            .map(|batch| batch.map(|b| b.num_rows()))
            .sum::<Result<_, _>>()?;
        assert_eq!(nested_rows, rows);
        Ok(())
    }
}
