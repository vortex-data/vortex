// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An IVF (inverted file) vector index for approximate top-k cosine similarity search.
//!
//! The build clusters every vector with spherical k-means into `lists` lists, and stores each
//! list's rows with int8-quantized copies of their vectors. A search scores the query against the
//! centroids, reads only the `nprobe` closest lists, and ranks their rows by the quantized
//! similarity.
//!
//! Index rows are one per list, sorted by list id, plus one directory row (list id
//! [`DIRECTORY_LIST`]) holding every centroid. A search reads the directory, then the probed lists,
//! and the sorted list column's zone map keeps every other list's segments unread.
//!
//! The results are approximate in two ways: a true neighbour can sit in a list that is not probed,
//! and scores come from quantized vectors. Probe more lists for better recall.

use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::or_collect;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_layout::layouts::indexed::IndexBuilder;
use vortex_layout::layouts::indexed::IndexId;
use vortex_layout::layouts::indexed::IndexQueryPlan;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::IndexVTableRef;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::postings::IndexRowsChunk;
use crate::postings::bitmap_from_bytes;
use crate::postings::bitmap_to_bytes;
use crate::postings::f32s_from_bytes;
use crate::postings::f32s_to_bytes;
use crate::rows::Hit;
use crate::rows::IndexRows;
use crate::rows::top_k;

/// Stable registry id of the IVF vector index.
pub const IVF_INDEX_ID: &str = "vortex.idx.ivf";

/// List id of the row holding every centroid.
pub const DIRECTORY_LIST: u32 = u32::MAX;

/// Lists probed by a search when the caller does not choose.
pub const DEFAULT_NPROBE: usize = 8;

const LIST_FIELD: &str = "list";
const CENTROIDS_FIELD: &str = "centroids";
const ROWS_FIELD: &str = "rows";
const CODES_FIELD: &str = "codes";

const KMEANS_ITERATIONS: usize = 25;
/// Training points per list; more are sampled evenly.
const KMEANS_SAMPLES_PER_LIST: usize = 512;

/// An IVF vector index over a column of `f32` vectors: either `FixedSizeList<f32>` or an
/// extension type stored as one, such as `vortex.tensor.vector`.
#[derive(Clone, Debug)]
pub struct IvfIndex;

impl IvfIndex {
    /// A shared handle to this index kind, for registration and write configuration.
    pub fn new_ref() -> IndexVTableRef {
        IndexVTableRef::new(Self)
    }
}

/// Options of the IVF index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IvfOptions {
    /// Number of lists. Zero picks about the square root of the row count. A built index records
    /// the number it used.
    pub lists: u32,
    /// Vector dimension, recorded by the build.
    pub dim: u32,
}

/// Index rows: list ids, each with its centroid run (only on the directory row), rows, and the
/// rows' quantized vectors.
#[derive(Debug)]
pub struct IvfEntries {
    lists: Vec<u32>,
    centroids: Vec<Vec<f32>>,
    rows: Vec<RoaringBitmap>,
    codes: Vec<ByteBuffer>,
}

fn index_fields() -> StructFields {
    let names: FieldNames = vec![LIST_FIELD, CENTROIDS_FIELD, ROWS_FIELD, CODES_FIELD].into();
    StructFields::new(
        names,
        vec![
            DType::Primitive(PType::U32, NonNullable),
            DType::Binary(NonNullable),
            DType::Binary(NonNullable),
            DType::Binary(NonNullable),
        ],
    )
}

/// The `f32` element type behind `dtype`, if it holds vectors of `f32`s.
fn vector_dim(dtype: &DType) -> Option<u32> {
    let storage = match dtype {
        DType::Extension(ext) => ext.storage_dtype(),
        other => other,
    };
    match storage {
        DType::FixedSizeList(element, dim, _)
            if matches!(element.as_ref(), DType::Primitive(PType::F32, _)) =>
        {
            Some(*dim)
        }
        _ => None,
    }
}

/// The vectors of `chunk` as one flat buffer, plus which rows are non-null.
fn flat_vectors(chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<(Vec<f32>, Vec<bool>)> {
    let storage = if matches!(chunk.dtype(), DType::Extension(_)) {
        chunk
            .clone()
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone()
    } else {
        chunk.clone()
    };
    let lists = storage.execute::<FixedSizeListArray>(ctx)?;
    let valid = lists.validity()?.execute_mask(lists.len(), ctx)?;
    let elements = lists.elements().clone().execute::<PrimitiveArray>(ctx)?;
    let valid = (0..lists.len()).map(|idx| valid.value(idx)).collect();
    Ok((elements.as_slice::<f32>().to_vec(), valid))
}

/// A dot product over eight lanes, which the compiler turns into SIMD.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (a_chunks, a_rest) = a.as_chunks::<8>();
    let (b_chunks, b_rest) = b.as_chunks::<8>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..8 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    let rest: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + rest
}

/// Scale `v` to unit length, leaving a zero vector as is.
fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Quantize a vector to `[scale: f32][dim x i8]`.
fn quantize(v: &[f32], out: &mut Vec<u8>) {
    let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
    out.extend_from_slice(&scale.to_le_bytes());
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is rounded and clamped to the i8 range first"
    )]
    out.extend(
        v.iter()
            .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8 as u8),
    );
}

/// A query quantized like the stored vectors, so it scores against their codes in integers.
struct QuantizedQuery {
    scale: f32,
    values: Vec<i8>,
}

impl QuantizedQuery {
    fn new(query: &[f32]) -> Self {
        let mut code = Vec::with_capacity(4 + query.len());
        quantize(query, &mut code);
        let (scale, values) = split_code(&code);
        Self {
            scale,
            values: values.iter().map(|&c| c as i8).collect(),
        }
    }

    /// The dot product with a vector quantized by [`quantize`].
    fn dot(&self, code: &[u8]) -> f32 {
        let (scale, values) = split_code(code);
        let mut lanes = [0i32; 16];
        let (q_chunks, q_rest) = self.values.as_chunks::<16>();
        let (c_chunks, c_rest) = values.as_chunks::<16>();
        for (q, c) in q_chunks.iter().zip(c_chunks) {
            for lane in 0..16 {
                lanes[lane] += i32::from(q[lane]) * i32::from(c[lane] as i8);
            }
        }
        let rest: i32 = q_rest
            .iter()
            .zip(c_rest)
            .map(|(&q, &c)| i32::from(q) * i32::from(c as i8))
            .sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "a sum of 127 * 127 products stays exact in f32 for any practical dimension"
        )]
        let sum = (lanes.iter().sum::<i32>() + rest) as f32;
        sum * scale * self.scale
    }
}

/// The scale and the `i8` values (as bytes) of a code written by [`quantize`].
fn split_code(code: &[u8]) -> (f32, &[u8]) {
    let scale = f32::from_le_bytes([code[0], code[1], code[2], code[3]]);
    (scale, &code[4..])
}

/// The index of the centroid nearest `v` by cosine similarity.
fn nearest(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(idx, c)| (idx, dot(v, c)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(idx, _)| idx)
}

/// Spherical k-means over unit vectors, seeded with evenly spaced points.
fn kmeans(vectors: &[f32], dim: usize, lists: usize) -> Vec<Vec<f32>> {
    let count = vectors.len() / dim;
    let at = |idx: usize| &vectors[idx * dim..(idx + 1) * dim];
    let samples: Vec<usize> = {
        let wanted = (lists * KMEANS_SAMPLES_PER_LIST).min(count);
        (0..wanted).map(|i| i * count / wanted).collect()
    };
    let mut centroids: Vec<Vec<f32>> = (0..lists)
        .map(|i| at(samples[i * samples.len() / lists]).to_vec())
        .collect();
    for _ in 0..KMEANS_ITERATIONS {
        let mut sums = vec![vec![0f32; dim]; lists];
        for &sample in &samples {
            let v = at(sample);
            let sum = &mut sums[nearest(v, &centroids)];
            sum.iter_mut().zip(v).for_each(|(s, x)| *s += x);
        }
        for (centroid, mut sum) in centroids.iter_mut().zip(sums) {
            // An emptied cluster keeps its previous centroid.
            if sum.iter().any(|x| *x != 0.0) {
                normalize(&mut sum);
                *centroid = sum;
            }
        }
    }
    centroids
}

impl IndexVTable for IvfIndex {
    type Options = IvfOptions;
    type Builder = IvfBuilder;
    /// Never constructed: this index ranks rows and does not answer filters.
    type Query = ();
    type Chunk = IvfEntries;

    fn id(&self) -> IndexId {
        static ID: CachedId = CachedId::new(IVF_INDEX_ID);
        *ID
    }

    fn serialize_options(&self, options: &IvfOptions) -> Vec<u8> {
        let mut bytes = options.lists.to_le_bytes().to_vec();
        bytes.extend_from_slice(&options.dim.to_le_bytes());
        bytes
    }

    fn deserialize_options(&self, options: &[u8]) -> VortexResult<IvfOptions> {
        if options.is_empty() {
            return Ok(IvfOptions::default());
        }
        vortex_ensure_eq!(options.len(), 8, "IVF options must be 8 bytes");
        Ok(IvfOptions {
            lists: u32::from_le_bytes([options[0], options[1], options[2], options[3]]),
            dim: u32::from_le_bytes([options[4], options[5], options[6], options[7]]),
        })
    }

    fn index_dtype(&self, dtype: &DType, _options: &IvfOptions) -> Option<DType> {
        vector_dim(dtype).map(|_| DType::Struct(index_fields(), NonNullable))
    }

    fn builder(
        &self,
        dtype: &DType,
        options: &IvfOptions,
        _data_block_len: Option<u64>,
        _session: &VortexSession,
    ) -> VortexResult<IvfBuilder> {
        let dim = vector_dim(dtype).ok_or_else(|| vortex_err!("IVF index cannot index {dtype}"))?;
        Ok(IvfBuilder {
            lists: options.lists,
            dim,
            vectors: Vec::new(),
            rows: Vec::new(),
        })
    }

    fn plan(
        &self,
        _expr: &BoundExpression,
        _dtype: &DType,
        _index_dtype: &DType,
        _options: &IvfOptions,
    ) -> VortexResult<Option<IndexQueryPlan<()>>> {
        // Approximate nearest neighbours can miss true matches, so this index never claims a
        // filter; it is only reached through `IndexRows::search_vectors`.
        Ok(None)
    }

    fn encode(
        &self,
        chunk: &IvfEntries,
        _options: &IvfOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let rows = chunk
            .rows
            .iter()
            .map(bitmap_to_bytes)
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new_with_dtype(
            vec![
                chunk
                    .lists
                    .iter()
                    .copied()
                    .collect::<PrimitiveArray>()
                    .into_array(),
                VarBinViewArray::from_iter_bin(chunk.centroids.iter().map(|c| f32s_to_bytes(c)))
                    .into_array(),
                VarBinViewArray::from_iter_bin(rows).into_array(),
                VarBinViewArray::from_iter_bin(&chunk.codes).into_array(),
            ],
            index_fields(),
            chunk.lists.len(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn decode(
        &self,
        chunk: ArrayRef,
        _options: &IvfOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<IvfEntries> {
        let entries = IndexRowsChunk::new(chunk, ctx)?;
        let lists = entries
            .field(LIST_FIELD)?
            .execute::<PrimitiveArray>(ctx)?
            .as_slice::<u32>()
            .to_vec();
        let centroids = entries
            .binary_field(CENTROIDS_FIELD, ctx)?
            .iter()
            .map(|bytes| f32s_from_bytes(bytes.as_slice()))
            .collect::<VortexResult<_>>()?;
        let rows = entries
            .binary_field(ROWS_FIELD, ctx)?
            .iter()
            .map(|bytes| bitmap_from_bytes(bytes.as_slice()))
            .collect::<VortexResult<_>>()?;
        let codes = entries.binary_field(CODES_FIELD, ctx)?;
        Ok(IvfEntries {
            lists,
            centroids,
            rows,
            codes,
        })
    }

    fn resolve(
        &self,
        _query: &(),
        _chunks: &[Arc<IvfEntries>],
        _data_row_count: u64,
        _options: &IvfOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<RowLocator> {
        vortex_bail!("The IVF index does not answer filters")
    }
}

impl IndexRows<IvfIndex> {
    /// The `k` rows whose vectors are most cosine-similar to `query`, best first, searching the
    /// `nprobe` lists whose centroids are closest to it.
    pub async fn search_vectors(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> VortexResult<Vec<Hit>> {
        let dim = self.options.dim as usize;
        vortex_ensure_eq!(query.len(), dim, "Query vector has the wrong dimension");
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut query = query.to_vec();
        normalize(&mut query);

        let centroids = self.centroids().await?;
        if centroids.is_empty() {
            return Ok(Vec::new());
        }

        let mut probes: Vec<(u32, f32)> = centroids
            .chunks_exact(dim)
            .enumerate()
            .map(|(list, centroid)| Ok((u32::try_from(list)?, dot(&query, centroid))))
            .collect::<VortexResult<_>>()?;
        probes.sort_by(|a, b| b.1.total_cmp(&a.1));
        probes.truncate(nprobe.max(1));

        let Some(filter) = or_collect(
            probes
                .iter()
                .map(|(list, _)| eq(col(LIST_FIELD), lit(*list))),
        ) else {
            return Ok(Vec::new());
        };
        let lists = self.chunks(Some(filter)).await?;

        let quantized = QuantizedQuery::new(&query);
        let code_len = 4 + dim;
        let mut hits = Vec::new();
        for entries in &lists {
            for (rows, codes) in entries.rows.iter().zip(&entries.codes) {
                for (row, code) in rows.iter().zip(codes.chunks_exact(code_len)) {
                    hits.push(Hit {
                        row: u64::from(row),
                        score: quantized.dot(code),
                    });
                }
            }
        }
        Ok(top_k(hits, k))
    }

    /// Every centroid, flattened: read from the directory row on the first search, then reused.
    async fn centroids(&self) -> VortexResult<Arc<Vec<f32>>> {
        if let Some(centroids) = self
            .memo
            .get()
            .and_then(|memo| Arc::clone(memo).downcast::<Vec<f32>>().ok())
        {
            return Ok(centroids);
        }
        let directory = self
            .chunks(Some(eq(col(LIST_FIELD), lit(DIRECTORY_LIST))))
            .await?;
        let centroids = Arc::new(
            directory
                .iter()
                .flat_map(|entries| entries.centroids.iter())
                .find(|centroids| !centroids.is_empty())
                .cloned()
                .unwrap_or_default(),
        );
        // A concurrent first search may have set it already; both read the same centroids.
        drop(self.memo.set(Arc::clone(&centroids) as _));
        Ok(centroids)
    }
}

/// Buffers every vector, then clusters and quantizes them on finish.
pub struct IvfBuilder {
    lists: u32,
    dim: u32,
    /// Unit vectors, flattened.
    vectors: Vec<f32>,
    rows: Vec<u32>,
}

impl IndexBuilder for IvfBuilder {
    type Options = IvfOptions;
    type Chunk = IvfEntries;

    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dim = self.dim as usize;
        let (vectors, valid) = flat_vectors(chunk, ctx)?;
        for (idx, valid) in valid.into_iter().enumerate() {
            if !valid {
                continue;
            }
            let mut v = vectors[idx * dim..(idx + 1) * dim].to_vec();
            normalize(&mut v);
            self.vectors.extend_from_slice(&v);
            self.rows.push(u32::try_from(row_offset + idx as u64)?);
        }
        Ok(())
    }

    fn finish(self) -> VortexResult<Option<(Vec<IvfEntries>, IvfOptions)>> {
        let dim = self.dim as usize;
        let count = self.rows.len();
        if count == 0 || dim == 0 {
            // Nothing to search; the index reads as absent.
            return Ok(None);
        }
        let lists = match self.lists {
            0 => count.isqrt().max(1),
            lists => lists as usize,
        }
        .clamp(1, count);
        let centroids = kmeans(&self.vectors, dim, lists);

        let mut members: Vec<(RoaringBitmap, Vec<u8>)> =
            vec![(RoaringBitmap::new(), Vec::new()); lists];
        // Rows arrive in order, so each list's bitmap iterates in the order its codes are pushed.
        for (idx, &row) in self.rows.iter().enumerate() {
            let v = &self.vectors[idx * dim..(idx + 1) * dim];
            let (rows, codes) = &mut members[nearest(v, &centroids)];
            rows.insert(row);
            quantize(v, codes);
        }

        let mut entries = IvfEntries {
            lists: Vec::with_capacity(lists + 1),
            centroids: Vec::with_capacity(lists + 1),
            rows: Vec::with_capacity(lists + 1),
            codes: Vec::with_capacity(lists + 1),
        };
        for (list, (rows, codes)) in members.into_iter().enumerate() {
            entries.lists.push(u32::try_from(list)?);
            entries.centroids.push(Vec::new());
            entries.rows.push(rows);
            entries.codes.push(ByteBuffer::from(codes));
        }
        entries.lists.push(DIRECTORY_LIST);
        entries.centroids.push(centroids.concat());
        entries.rows.push(RoaringBitmap::new());
        entries.codes.push(ByteBuffer::empty());

        Ok(Some((
            vec![entries],
            IvfOptions {
                lists: u32::try_from(lists)?,
                dim: self.dim,
            },
        )))
    }

    fn buffered_bytes(&self) -> u64 {
        (self.vectors.len() * 4 + self.rows.len() * 4) as u64
    }
}
