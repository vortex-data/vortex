// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DataFusion tables for VBench cosine top-k queries, with optional price prefiltering.
//! Register either or both engines in the same session. Both expose the recipe fields
//! `id: UInt64`, `emb: FixedSizeList<Float32>`, and `price: Int64`, all non-null.
//! [`register_topk`] exposes indexed retrieval as an explicit SQL table function.
//! The continuous benchmark uses this function with the default DataFusion planner.
//! Calling [`enable_ann`] opts these tables into approximate selection for the SQL shape below.
//!
//! ```sql
//! SELECT id, cosine_distance(emb, $1) AS distance
//! FROM recipes WHERE price >= 90 ORDER BY distance LIMIT 10
//! ```
//!
//! Bind `$1` to a list of floats with the same dimension as `emb`. Omit the predicate for Q1.
//! Other query shapes use a full scan. Both paths return distances computed by DataFusion.

mod planner;
mod table;
mod topk;

use std::sync::Arc;

use anyhow::ensure;
use datafusion::prelude::SessionContext;
use lance::Dataset;
use lance::deps::arrow_schema as ls;
pub use topk::register_topk;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability::NonNullable;
use vortex::array::dtype::PType;
use vortex::file::VortexFile;
use vortex_index::IndexFile;
use vortex_index_ivf::IvfIndex;

use self::planner::VBenchAnn;
use self::table::Backend;
use self::table::RecipeTable;

/// Enable the VBench ANN rewrite for this session's explicitly registered recipe tables.
pub fn enable_ann(ctx: &SessionContext) {
    ctx.add_analyzer_rule(Arc::new(VBenchAnn));
}

/// Register a Vortex recipe file together with the single-file cosine index built on its `emb`.
pub fn register_vortex(
    ctx: &SessionContext,
    name: &str,
    file: VortexFile,
    index: &IndexFile<IvfIndex>,
    nprobes: usize,
) -> anyhow::Result<()> {
    ensure!(nprobes > 0, "nprobes must be positive");
    ensure!(
        index.column() == "emb"
            && index.files().len() == 1
            && index.rows().data_row_count() == file.row_count(),
        "expected the single-file IVF index for this file's emb column"
    );
    let fields = file
        .dtype()
        .as_struct_fields_opt()
        .ok_or_else(|| anyhow::anyhow!("expected a recipe struct"))?;
    ensure!(
        fields.field("id") == Some(DType::Primitive(PType::U64, NonNullable))
            && fields.field("price") == Some(DType::Primitive(PType::I64, NonNullable)),
        "expected non-null UInt64 id and Int64 price"
    );
    let Some(DType::FixedSizeList(item, dim, NonNullable)) = fields.field("emb") else {
        anyhow::bail!("expected a non-null fixed-size embedding");
    };
    ensure!(
        *item == DType::Primitive(PType::F32, NonNullable)
            && fields.field("emb").as_ref() == Some(index.rows().source_dtype()),
        "expected non-null Float32 embeddings matching the index"
    );
    let backend = Backend::Vortex {
        file: Arc::new(file),
        index: Arc::clone(index.rows()),
    };
    ctx.register_table(
        name,
        Arc::new(RecipeTable::new(backend, i32::try_from(dim)?, nprobes)),
    )?;
    Ok(())
}

/// Register a Lance recipe dataset with its existing cosine IVF index on `emb`.
pub fn register_lance(
    ctx: &SessionContext,
    name: &str,
    dataset: Arc<Dataset>,
    nprobes: usize,
) -> anyhow::Result<()> {
    ensure!(nprobes > 0, "nprobes must be positive");
    let schema = ls::Schema::from(dataset.schema());
    for (name, ty) in [("id", ls::DataType::UInt64), ("price", ls::DataType::Int64)] {
        let field = schema.field_with_name(name)?;
        ensure!(
            !field.is_nullable() && field.data_type() == &ty,
            "expected non-null {ty} {name}"
        );
    }
    let emb = schema.field_with_name("emb")?;
    let ls::DataType::FixedSizeList(item, dim) = emb.data_type() else {
        anyhow::bail!("expected fixed-size embeddings");
    };
    ensure!(
        !emb.is_nullable() && item.data_type() == &ls::DataType::Float32,
        "expected non-null Float32 embeddings"
    );
    ctx.register_table(
        name,
        Arc::new(RecipeTable::new(Backend::Lance(dataset), *dim, nprobes)),
    )?;
    Ok(())
}
