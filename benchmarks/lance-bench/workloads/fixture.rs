// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are 32 or 128 and the generator returns at most 31 bits"
)]

use std::sync::Arc;

use lance::Dataset;
use lance::dataset::WriteParams;
use lance::deps::arrow_array as la;
use lance::deps::arrow_schema as ls;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_index::IndexType;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability::NonNullable;
use vortex::array::dtype::PType;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::file::VortexFile;
use vortex::session::VortexSession;
use vortex_index::IndexFileBuilder;
use vortex_index_ivf::IvfIndex;
use vortex_index_ivf::IvfOptions;

pub(crate) const K: usize = 10;
/// How many random queries, besides the benchmark's own, the recall check runs.
const RECALL_QUERIES: usize = 20;

/// The table's size and the IVF's list count.
#[derive(Clone, Copy)]
pub(crate) struct Scale {
    pub(crate) label: &'static str,
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) lists: u32,
}

impl Scale {
    pub(crate) const LARGE: Self = Self {
        label: "400k",
        rows: 400_000,
        dim: 128,
        lists: 256,
    };
    pub(crate) const SMALL: Self = Self {
        label: "20k",
        rows: 20_000,
        dim: 32,
        lists: 64,
    };
}

/// A small deterministic generator, so both sides get the same data.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
}

/// The table: `id` and an embedding near one of a few topics.
pub(crate) struct Data {
    pub(crate) scale: Scale,
    vectors: Vec<f32>,
    pub(crate) query: Vec<f32>,
}

impl Data {
    pub(crate) fn generate(scale: Scale) -> Self {
        let Scale { rows, dim, .. } = scale;
        let mut rng = Lcg(42);
        let topics: Vec<f32> = (0..16 * dim).map(|_| rng.unit() - 0.5).collect();
        let mut vectors = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let topic = (rng.next() % 16) as usize;
            vectors.extend(
                topics[topic * dim..(topic + 1) * dim]
                    .iter()
                    .map(|t| t + (rng.unit() - 0.5) * 0.2),
            );
        }
        let query = vectors[..dim].iter().map(|v| v + 0.01).collect();
        Self {
            scale,
            vectors,
            query,
        }
    }

    fn prices(&self) -> impl Iterator<Item = i64> {
        (0..self.scale.rows).map(price)
    }

    pub(crate) fn vortex_array(&self) -> VortexResult<ArrayRef> {
        let ids: PrimitiveArray = (0..self.scale.rows as u64).collect();
        let elements = PrimitiveArray::new(Buffer::copy_from(&self.vectors), Validity::NonNullable);
        let vectors = FixedSizeListArray::new(
            elements.into_array(),
            self.scale.dim as u32,
            Validity::NonNullable,
            self.scale.rows,
        );
        Ok(StructArray::from_fields(&[
            ("id", ids.into_array()),
            ("emb", vectors.into_array()),
            (
                "price",
                self.prices().collect::<PrimitiveArray>().into_array(),
            ),
        ])?
        .into_array())
    }

    fn lance_batch(&self) -> lance::Result<la::RecordBatch> {
        let item = Arc::new(ls::Field::new("item", ls::DataType::Float32, false));
        let schema = Arc::new(ls::Schema::new(vec![
            ls::Field::new("id", ls::DataType::UInt64, false),
            ls::Field::new(
                "emb",
                ls::DataType::FixedSizeList(Arc::clone(&item), self.scale.dim as i32),
                false,
            ),
            ls::Field::new("price", ls::DataType::Int64, false),
        ]));
        let vectors = la::FixedSizeListArray::new(
            item,
            self.scale.dim as i32,
            Arc::new(la::Float32Array::from(self.vectors.clone())),
            None,
        );
        Ok(la::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(la::UInt64Array::from_iter_values(0..self.scale.rows as u64)),
                Arc::new(vectors),
                Arc::new(la::Int64Array::from_iter_values(self.prices())),
            ],
        )?)
    }

    /// The benchmark's cosine query, then [`RECALL_QUERIES`] more near random rows.
    pub(crate) fn vector_queries(&self) -> Vec<Vec<f32>> {
        let dim = self.scale.dim;
        let mut rng = Lcg(7);
        let mut queries = vec![self.query.clone()];
        for _ in 0..RECALL_QUERIES {
            let row = (rng.next() as usize) % self.scale.rows;
            queries.push(
                self.vectors[row * dim..(row + 1) * dim]
                    .iter()
                    .map(|v| v + (rng.unit() - 0.5) * 0.2)
                    .collect(),
            );
        }
        queries
    }
}

// Use a different sequence from the embedding generator so price is independent of topic.
pub(crate) fn price(row: usize) -> i64 {
    ((row * 73 + 17) % 100) as i64
}

/// Exact cosine top-[`K`] rows, best first, ties by row.
pub(crate) fn exact_cosine(data: &Data, query: &[f32], min_price: Option<i64>) -> Vec<u64> {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let query_norm = norm(query);
    let mut scored: Vec<(u64, f32)> = data
        .vectors
        .chunks_exact(data.scale.dim)
        .enumerate()
        .filter(|(row, _)| min_price.is_none_or(|floor| price(*row) >= floor))
        .map(|(row, v)| {
            let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
            (row as u64, dot / (norm(v) * query_norm))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().take(K).map(|(row, _)| row).collect()
}

pub(crate) async fn lance_dataset(uri: &str, data: &Data) -> lance::Result<Dataset> {
    let batch = data.lance_batch()?;
    let reader = la::RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader, uri, Some(WriteParams::default())).await?;
    let params = VectorIndexParams::with_ivf_sq_params(
        MetricType::Cosine,
        IvfBuildParams::new(data.scale.lists as usize),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["emb"], IndexType::Vector, None, &params, true)
        .await?;
    Ok(dataset)
}

pub(crate) async fn build_ivf(
    session: &VortexSession,
    file: &VortexFile,
    scale: Scale,
) -> VortexResult<IndexFileBuilder<IvfIndex>> {
    let vector_dtype = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::F32, NonNullable)),
        scale.dim as u32,
        NonNullable,
    );
    let mut ivf = IndexFileBuilder::new(
        IvfIndex,
        IvfOptions {
            lists: scale.lists,
            ..Default::default()
        },
        "emb",
        vector_dtype,
        session,
    )?;
    ivf.add_file("data.vortex", file).await?;
    Ok(ivf)
}
