// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JSONBench queries over Vortex files with Variant `data` columns.
//!
//! The Vortex scan evaluates both the query's filter and the extraction of the paths it
//! aggregates: `variant_get` expressions on the `data` column. Each scan split aggregates its rows
//! into a [`Partial`] on the scan's own worker task.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use futures::StreamExt;
use futures::TryStreamExt;
use vortex::array::VortexSessionExecute;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::eq;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::or;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::variant_get;
use vortex::file::OpenOptionsSessionExt;
use vortex::scalar_fn::fns::variant_get::VariantPath;
use vortex::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_arrow::ArrowSessionExt;
use vortex_bench::SESSION;

use crate::agg::Partial;
use crate::agg::Query;
use crate::agg::Row;
use crate::agg::i64_at;
use crate::agg::str_at;

/// `variant_get(data, path)` as a nullable string or 64-bit integer.
fn data_path(path: &str, dtype: DType) -> Expression {
    variant_get(
        get_item("data", root()),
        VariantPath::new(path.split('.').map(VariantPathElement::field)),
        Some(dtype),
    )
}

fn str_path(path: &str) -> Expression {
    data_path(path, DType::Utf8(Nullability::Nullable))
}

fn filter(query: Query) -> Option<Expression> {
    let mut conjuncts = Vec::new();
    if query.filters_commit_creates() {
        conjuncts.push(eq(str_path("kind"), lit("commit")));
        conjuncts.push(eq(str_path("commit.operation"), lit("create")));
    }
    if let Some(collection_filter) = query.collection_filter().and_then(|collections| {
        collections
            .iter()
            .map(|collection| eq(str_path("commit.collection"), lit(*collection)))
            .reduce(or)
    }) {
        conjuncts.push(collection_filter);
    }
    conjuncts.into_iter().reduce(and)
}

/// The scan projection and the Arrow fields its columns are exported as.
fn projection(query: Query) -> (Expression, Vec<Field>) {
    let outputs = query.outputs();
    let mut columns = Vec::new();
    let mut fields = Vec::new();
    if outputs.collection {
        columns.push(("collection", str_path("commit.collection")));
        fields.push(Field::new("collection", DataType::Utf8View, true));
    }
    if outputs.did {
        columns.push(("did", str_path("did")));
        fields.push(Field::new("did", DataType::Utf8View, true));
    }
    if outputs.time_us {
        columns.push((
            "time_us",
            data_path(
                "time_us",
                DType::Primitive(vortex::dtype::PType::I64, Nullability::Nullable),
            ),
        ));
        fields.push(Field::new("time_us", DataType::Int64, true));
    }
    (pack(columns, Nullability::NonNullable), fields)
}

/// Aggregate the exported rows of one scan split.
fn aggregate(query: Query, columns: &arrow_array::StructArray) -> Partial {
    let column = |name: &str| columns.column_by_name(name);
    let collection = column("collection").map(|c| c.as_string_view());
    let did = column("did").map(|c| c.as_string_view());
    let time_us = column("time_us").map(|c| c.as_primitive::<arrow_array::types::Int64Type>());

    let mut partial = Partial::new(query);
    for idx in 0..columns.len() {
        partial.add(Row {
            collection: str_at(collection, idx),
            did: str_at(did, idx),
            time_us: i64_at(time_us, idx),
        });
    }
    partial
}

pub async fn run(query: Query, files: Vec<PathBuf>) -> anyhow::Result<Partial> {
    let filter = filter(query);
    let (projection, fields) = projection(query);
    let struct_field = Arc::new(Field::new_struct("", fields, false));

    let n_files = files.len().max(1);
    let partials = futures::stream::iter(files)
        .map(|path| {
            let filter = filter.clone();
            let projection = projection.clone();
            let struct_field = Arc::clone(&struct_field);
            async move {
                let file = SESSION.open_options().open_path(&path).await?;
                let dtype = file.dtype().clone();
                let scan = file
                    .scan()?
                    .with_projection(projection.bind(&dtype)?)
                    .with_some_filter(filter.map(|filter| filter.bind(&dtype)).transpose()?)
                    .map(move |chunk| {
                        let mut ctx = SESSION.create_execution_ctx();
                        let arrow =
                            SESSION
                                .arrow()
                                .execute_arrow(chunk, Some(&struct_field), &mut ctx)?;
                        Ok(aggregate(query, arrow.as_struct()))
                    });
                let partial = scan
                    .into_stream()?
                    .try_fold(Partial::new(query), |mut total, partial| async move {
                        total.merge(partial);
                        Ok(total)
                    })
                    .await?;
                anyhow::Ok(partial)
            }
        })
        .buffer_unordered(n_files);

    partials
        .try_fold(Partial::new(query), |mut total, partial| async move {
            total.merge(partial);
            Ok(total)
        })
        .await
}
