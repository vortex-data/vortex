// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A full-text inverted index: for each token, the rows holding it and how often.
//!
//! Answers [`text_match`](crate::text_match::text_match) exactly, since it tokenizes the same way,
//! and ranks rows for a free-text query with BM25 through [`IndexRows::search_bm25`].
//!
//! Postings are plain Vortex columns, so the writer's compressor packs them: each term's postings
//! are a `List<Struct{gap, tf}>` of the gap from the previous row and the term's frequency there,
//! which bit-pack to a few bits a posting. A bit-packed chunk has one bit width, so terms are
//! ordered by how many rows hold them (in powers of two), then alphabetically: each chunk of
//! terms then has gaps of about the same size. Zone maps over the term still prune a lookup to
//! one chunk per frequency class.
//!
//! The index is built once over all the rows it covers, so each term also stores a bound on the
//! largest BM25 score it gives any row, which a search uses to skip scoring the rows of terms too
//! common to change its top `k` (MaxScore). Row lengths, which BM25 normalizes by, are stored
//! once per row rather than per posting: as the postings of the empty term, which no text
//! tokenizes to, with each row's length in place of its frequency.

use std::cmp::Reverse;
use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::or_collect;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_layout::layouts::indexed::IndexBuilder;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexId;
use vortex_layout::layouts::indexed::IndexQueryPlan;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::IndexVTableRef;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_map::HashMap;

use crate::postings::IndexRowsChunk;
use crate::rows::Hit;
use crate::rows::IndexRows;
use crate::rows::top_k;
use crate::text_match::TextMatch;
use crate::tokenize::tokenize;

/// Stable registry id of the full-text index.
pub const INVERTED_INDEX_ID: &str = "vortex.idx.inverted";

const TERM_FIELD: &str = "term";
const POSTINGS_FIELD: &str = "postings";
const GAP_FIELD: &str = "gap";
const TF_FIELD: &str = "tf";
const BOUND_FIELD: &str = "bound";

/// The term whose postings hold every non-empty row's length instead of a frequency.
const CORPUS_TERM: &str = "";

/// BM25 term-frequency saturation.
const BM25_K1: f32 = 1.2;
/// BM25 document-length normalization.
const BM25_B: f32 = 0.75;
/// A term's stored bound is its largest `tf / (tf + norm)`, rounded up to this many steps.
const BOUND_STEPS: f32 = 255.0;

/// A full-text inverted index over a UTF-8 column.
#[derive(Clone, Debug)]
pub struct InvertedIndex;

impl InvertedIndex {
    /// A shared handle to this index kind, for registration and write configuration.
    pub fn new_ref() -> IndexVTableRef {
        IndexVTableRef::new(Self)
    }
}

/// Terms, each with its postings still encoded: a lookup decodes just the terms it reads.
#[derive(Debug)]
pub struct InvertedEntries {
    terms: Vec<String>,
    /// `List<Struct{gap: u32, tf: u32}>`, one list per term, in row order.
    postings: ListViewArray,
    bounds: Vec<u8>,
}

/// One term's postings, decoded.
struct Posting {
    rows: Vec<u32>,
    tfs: Buffer<u32>,
    bound: u8,
}

/// What BM25 needs of the whole corpus: its size and each row's length normalization.
struct Corpus {
    docs: f32,
    /// `k1 * (1 - b + b * length / average)`, by row. Rows past the end are empty.
    norms: Vec<f32>,
}

impl Corpus {
    /// From the non-empty rows, ascending, and their lengths.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "corpus counts only weight scores, so f32 precision is ample"
    )]
    fn new(rows: &[u32], lengths: &[u32]) -> Self {
        let total: u64 = lengths.iter().map(|&length| u64::from(length)).sum();
        let avg_length = if rows.is_empty() {
            1.0
        } else {
            (total as f64 / rows.len() as f64) as f32
        };
        let norm_a = BM25_K1 * (1.0 - BM25_B);
        let norm_b = BM25_K1 * BM25_B / avg_length;
        let mut norms = vec![norm_a; rows.last().map_or(0, |&row| row as usize + 1)];
        for (&row, &length) in rows.iter().zip(lengths) {
            norms[row as usize] = norm_a + norm_b * length as f32;
        }
        Self {
            docs: rows.len() as f32,
            norms,
        }
    }

    /// IDF times `k1 + 1`, for a term held by `df` rows.
    #[expect(clippy::cast_precision_loss, reason = "row counts only weight scores")]
    fn weight(&self, df: usize) -> f32 {
        let df = df as f32;
        (1.0 + (self.docs - df + 0.5) / (df + 0.5)).ln() * (BM25_K1 + 1.0)
    }

    /// `tf / (tf + norm)`: a term's score in `row` is this times its weight.
    #[expect(
        clippy::cast_precision_loss,
        reason = "term frequencies only weight scores"
    )]
    fn saturation(&self, tf: u32, row: u32) -> f32 {
        let tf = tf as f32;
        tf / (tf + self.norms[row as usize])
    }

    /// The bound stored for postings `(row, tf)`: their largest saturation, rounded up.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a saturation is in [0, 1), so its steps fit a u8"
    )]
    fn bound(&self, postings: impl Iterator<Item = (u32, u32)>) -> u8 {
        let max = postings
            .map(|(row, tf)| self.saturation(tf, row))
            .fold(0.0, f32::max);
        (max * BOUND_STEPS).ceil().clamp(0.0, BOUND_STEPS) as u8
    }
}

/// A posting list being scored: its rows and frequencies, the term's weight, and the most it
/// adds to a row.
struct Scoring {
    posting: Posting,
    weight: f32,
    max_score: f32,
}

impl Scoring {
    fn new(posting: Posting, corpus: &Corpus) -> Self {
        let weight = corpus.weight(posting.rows.len());
        let max_score = weight * f32::from(posting.bound) / BOUND_STEPS;
        Self {
            posting,
            weight,
            max_score,
        }
    }

    /// Each row and the term's score there, in row order.
    fn scored<'a>(&'a self, corpus: &'a Corpus) -> impl Iterator<Item = (u32, f32)> + 'a {
        self.posting
            .rows
            .iter()
            .zip(self.posting.tfs.iter())
            .map(|(&row, &tf)| (row, self.weight * corpus.saturation(tf, row)))
    }

    /// The term's score in `row`, if the row holds the term.
    fn score_of(&self, row: u32, corpus: &Corpus) -> Option<f32> {
        let idx = self.posting.rows.binary_search(&row).ok()?;
        Some(self.weight * corpus.saturation(self.posting.tfs[idx], row))
    }
}

fn u32_field(
    postings: &StructArray,
    name: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<u32>> {
    Ok(postings
        .unmasked_field_by_name(name)?
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<u32>())
}

impl InvertedEntries {
    /// Terms are ordered by frequency class, so not sorted within a chunk.
    fn position(&self, term: &str) -> Option<usize> {
        self.terms.iter().position(|probe| probe == term)
    }

    fn get(&self, term: &str, ctx: &mut ExecutionCtx) -> VortexResult<Option<Posting>> {
        let Some(idx) = self.position(term) else {
            return Ok(None);
        };
        let postings = self
            .postings
            .list_elements_at(idx, ctx)?
            .execute::<StructArray>(ctx)?;
        let mut rows = u32_field(&postings, GAP_FIELD, ctx)?.to_vec();
        let mut row = 0u32;
        for gap in &mut rows {
            row = row.wrapping_add(*gap);
            *gap = row;
        }
        let tfs = u32_field(&postings, TF_FIELD, ctx)?;
        vortex_ensure_eq!(
            rows.len(),
            tfs.len(),
            "Inverted index entry has mismatched rows and frequencies"
        );
        Ok(Some(Posting {
            rows,
            tfs,
            bound: self.bounds[idx],
        }))
    }
}

fn find(
    chunks: &[Arc<InvertedEntries>],
    term: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Posting>> {
    for chunk in chunks {
        if let Some(posting) = chunk.get(term, ctx)? {
            return Ok(Some(posting));
        }
    }
    Ok(None)
}

fn posting_fields() -> StructFields {
    let names: FieldNames = vec![GAP_FIELD, TF_FIELD].into();
    let u32 = DType::Primitive(PType::U32, NonNullable);
    StructFields::new(names, vec![u32.clone(), u32])
}

fn index_fields() -> StructFields {
    let names: FieldNames = vec![TERM_FIELD, POSTINGS_FIELD, BOUND_FIELD].into();
    StructFields::new(
        names,
        vec![
            DType::Utf8(NonNullable),
            DType::List(
                Arc::new(DType::Struct(posting_fields(), NonNullable)),
                NonNullable,
            ),
            DType::Primitive(PType::U8, NonNullable),
        ],
    )
}

/// Distinct query tokens, sorted.
fn query_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = tokenize(query).collect();
    terms.sort_unstable();
    terms.dedup();
    terms
}

fn terms_filter(terms: impl IntoIterator<Item = String>) -> Option<Expression> {
    or_collect(
        terms
            .into_iter()
            .map(|term| eq(col(TERM_FIELD), lit(term.as_str()))),
    )
}

impl IndexVTable for InvertedIndex {
    type Options = ();
    type Builder = InvertedBuilder;
    /// Distinct query terms, all of which a matching row must contain.
    type Query = Vec<String>;
    type Chunk = InvertedEntries;

    fn id(&self) -> IndexId {
        static ID: CachedId = CachedId::new(INVERTED_INDEX_ID);
        *ID
    }

    fn serialize_options(&self, _options: &()) -> Vec<u8> {
        vec![]
    }

    fn deserialize_options(&self, _options: &[u8]) -> VortexResult<()> {
        Ok(())
    }

    fn index_dtype(&self, dtype: &DType, _options: &()) -> Option<DType> {
        dtype
            .is_utf8()
            .then(|| DType::Struct(index_fields(), NonNullable))
    }

    fn builder(
        &self,
        _dtype: &DType,
        _options: &(),
        _data_block_len: Option<u64>,
        session: &VortexSession,
    ) -> VortexResult<InvertedBuilder> {
        Ok(InvertedBuilder {
            terms: HashMap::new(),
            session: session.clone(),
        })
    }

    fn plan(
        &self,
        expr: &BoundExpression,
        _dtype: &DType,
        index_dtype: &DType,
        _options: &(),
    ) -> VortexResult<Option<IndexQueryPlan<Vec<String>>>> {
        if !expr.is::<TextMatch>() || !expr.child(0).is_root() || !expr.child(1).is::<Literal>() {
            return Ok(None);
        }
        let Some(query) = expr.child(1).as_::<Literal>().as_utf8_opt() else {
            return Ok(None);
        };
        let Some(query) = query.value() else {
            return Ok(None);
        };
        let terms = query_terms(query.as_str());
        // A query without tokens matches every non-null row, which a scan answers as cheaply.
        let Some(filter) = terms_filter(terms.iter().cloned()) else {
            return Ok(None);
        };
        Ok(Some(IndexQueryPlan {
            exactness: IndexExactness::Exact,
            filter: filter.bind(index_dtype)?,
            query: terms,
        }))
    }

    fn encode(
        &self,
        chunk: &InvertedEntries,
        _options: &(),
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(StructArray::try_new_with_dtype(
            vec![
                VarBinViewArray::from_iter_str(&chunk.terms).into_array(),
                chunk.postings.clone().into_array(),
                chunk
                    .bounds
                    .iter()
                    .copied()
                    .collect::<PrimitiveArray>()
                    .into_array(),
            ],
            index_fields(),
            chunk.terms.len(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn decode(
        &self,
        chunk: ArrayRef,
        _options: &(),
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<InvertedEntries> {
        let rows = IndexRowsChunk::new(chunk, ctx)?;
        let terms = rows
            .binary_field(TERM_FIELD, ctx)?
            .into_iter()
            .map(|bytes| {
                String::from_utf8(bytes.as_slice().to_vec())
                    .map_err(|err| vortex_err!("Inverted index term is not UTF-8: {err}"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let bounds = rows
            .field(BOUND_FIELD)?
            .execute::<PrimitiveArray>(ctx)?
            .as_slice::<u8>()
            .to_vec();
        debug_assert_eq!(rows.len(), terms.len());
        Ok(InvertedEntries {
            terms,
            postings: rows.field(POSTINGS_FIELD)?.execute::<ListViewArray>(ctx)?,
            bounds,
        })
    }

    fn resolve(
        &self,
        terms: &Vec<String>,
        chunks: &[Arc<InvertedEntries>],
        _data_row_count: u64,
        _options: &(),
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RowLocator> {
        let mut result: Option<RoaringBitmap> = None;
        for term in terms {
            let Some(posting) = find(chunks, term, ctx)? else {
                return Ok(RowLocator::empty_rows());
            };
            let rows = RoaringBitmap::from_sorted_iter(posting.rows.iter().copied())
                .map_err(|err| vortex_err!("Inverted index rows are not sorted: {err}"))?;
            result = Some(match result {
                None => rows,
                Some(acc) => acc & rows,
            });
        }
        Ok(RowLocator::Rows(result.unwrap_or_default()))
    }
}

/// Merge two lists of `(row, score)` sorted by row, summing the scores of rows in both.
fn merge(a: Vec<(u32, f32)>, b: impl Iterator<Item = (u32, f32)>) -> Vec<(u32, f32)> {
    let mut out = Vec::with_capacity(a.len());
    let mut a = a.into_iter().peekable();
    for (row, score) in b {
        while let Some(&(next, next_score)) = a.peek() {
            if next >= row {
                break;
            }
            out.push((next, next_score));
            a.next();
        }
        match a.peek() {
            Some(&(next, next_score)) if next == row => {
                out.push((row, next_score + score));
                a.next();
            }
            _ => out.push((row, score)),
        }
    }
    out.extend(a);
    out
}

/// The `k`th best score of `scored`, or `None` if it holds fewer than `k` rows.
fn kth_best(scored: &[(u32, f32)], k: usize) -> Option<f32> {
    if k == 0 || scored.len() < k {
        return None;
    }
    let mut scores: Vec<f32> = scored.iter().map(|(_, score)| *score).collect();
    let (_, kth, _) = scores.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    Some(*kth)
}

impl IndexRows<InvertedIndex> {
    /// The corpus statistics, read from the corpus entry on first use and then kept.
    async fn corpus(&self) -> VortexResult<Arc<Corpus>> {
        if let Some(corpus) = self.memo_get::<Corpus>() {
            return Ok(corpus);
        }
        let chunks = self.chunks_for_keys(&[CORPUS_TERM.to_string()]).await?;
        let mut ctx = self.session.create_execution_ctx();
        let lengths = find(&chunks, CORPUS_TERM, &mut ctx)?
            .ok_or_else(|| vortex_err!("Inverted index has no row lengths"))?;
        let corpus = Arc::new(Corpus::new(&lengths.rows, &lengths.tfs));
        self.memo_set(Arc::clone(&corpus));
        Ok(corpus)
    }

    /// The `k` rows that score best for `query` under BM25, best first.
    ///
    /// A row scores if it holds any query token. Only the entries for the query's tokens are
    /// read. Terms are scored in order of the most they can add to a row; once the terms left
    /// could not lift a row the earlier terms missed into the top `k`, they only add to the rows
    /// already found, looking each one up instead of walking their postings.
    pub async fn search_bm25(&self, query: &str, k: usize) -> VortexResult<Vec<Hit>> {
        let terms = query_terms(query);
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let corpus = self.corpus().await?;
        let chunks = self.chunks_for_keys(&terms).await?;
        let mut ctx = self.session.create_execution_ctx();
        let mut lists = Vec::with_capacity(terms.len());
        for term in &terms {
            if let Some(posting) = find(&chunks, term, &mut ctx)? {
                lists.push(Scoring::new(posting, &corpus));
            }
        }
        lists.sort_by(|a, b| b.max_score.total_cmp(&a.max_score));
        // What the terms from each position on can add to one row at most. Summed in f64 with
        // some slack, so f32 rounding of a row's own sum never exceeds it.
        let mut remaining = vec![0f64; lists.len() + 1];
        for idx in (0..lists.len()).rev() {
            remaining[idx] = remaining[idx + 1] + f64::from(lists[idx].max_score);
        }
        let bound = |idx: usize| remaining[idx] * (1.0 + 1e-5) + 1e-6;

        let mut scored: Vec<(u32, f32)> = Vec::new();
        let mut walked = 0;
        while walked < lists.len() {
            if kth_best(&scored, k).is_some_and(|kth| bound(walked) < f64::from(kth)) {
                break;
            }
            scored = merge(scored, lists[walked].scored(&corpus));
            walked += 1;
        }
        if walked < lists.len() {
            let floor = kth_best(&scored, k).map_or(f64::NEG_INFINITY, f64::from);
            let rest = bound(walked);
            scored.retain_mut(|(row, score)| {
                if f64::from(*score) + rest < floor {
                    return false;
                }
                for list in &lists[walked..] {
                    if let Some(extra) = list.score_of(*row, &corpus) {
                        *score += extra;
                    }
                }
                true
            });
        }

        let hits = scored
            .into_iter()
            .map(|(row, score)| Hit {
                row: u64::from(row),
                score,
            })
            .collect();
        Ok(top_k(hits, k))
    }
}

/// Builds an [`InvertedIndex`] while the column streams past.
pub struct InvertedBuilder {
    /// For each term, `(row, frequency)` in row order. Rows arrive in order, so appending keeps
    /// them sorted. The corpus term holds each non-empty row's length.
    terms: HashMap<String, Vec<(u32, u32)>>,
    session: VortexSession,
}

/// Concatenate each term's postings into one `List<Struct{gap, tf}>`.
fn posting_lists<'a>(
    lists: impl IntoIterator<Item = &'a [(u32, u32)]>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    let mut gaps = Vec::new();
    let mut tfs = Vec::new();
    let mut offsets = vec![0u64];
    for list in lists {
        let mut prev = 0;
        for &(row, tf) in list {
            gaps.push(row - prev);
            tfs.push(tf);
            prev = row;
        }
        offsets.push(gaps.len() as u64);
    }
    let len = gaps.len();
    let elements = StructArray::try_new_with_dtype(
        vec![
            Buffer::from(gaps).into_array(),
            Buffer::from(tfs).into_array(),
        ],
        posting_fields(),
        len,
        Validity::NonNullable,
    )?;
    ListArray::try_new(
        elements.into_array(),
        Buffer::from(offsets).into_array(),
        Validity::NonNullable,
        ctx,
    )?
    .into_array()
    .execute::<ListViewArray>(ctx)
}

impl IndexBuilder for InvertedBuilder {
    type Options = ();
    type Chunk = InvertedEntries;

    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let values = chunk.clone().execute::<VarBinViewArray>(ctx)?;
        let valid = values.validity()?.execute_mask(values.len(), ctx)?;
        let mut row_terms: HashMap<String, u32> = HashMap::new();
        for idx in 0..values.len() {
            if !valid.value(idx) {
                continue;
            }
            let row = u32::try_from(row_offset + idx as u64)?;
            let bytes = values.bytes_at(idx);
            let text = String::from_utf8_lossy(bytes.as_slice());
            let mut length = 0u32;
            row_terms.clear();
            for token in tokenize(&text) {
                length += 1;
                *row_terms.entry(token).or_default() += 1;
            }
            if length == 0 {
                continue;
            }
            for (term, tf) in row_terms.drain() {
                self.terms.entry(term).or_default().push((row, tf));
            }
            self.terms
                .entry(CORPUS_TERM.to_string())
                .or_default()
                .push((row, length));
        }
        Ok(())
    }

    fn finish(mut self) -> VortexResult<Option<(Vec<InvertedEntries>, ())>> {
        let mut ctx = self.session.create_execution_ctx();
        let lengths = self.terms.remove(CORPUS_TERM).unwrap_or_default();
        let (rows, row_lengths): (Vec<u32>, Vec<u32>) = lengths.iter().copied().unzip();
        let corpus = Corpus::new(&rows, &row_lengths);

        // The corpus entry first, then by frequency class, most common first, then by term.
        let mut entries: Vec<_> = self.terms.into_iter().collect();
        entries.sort_unstable_by(|(a, pa), (b, pb)| {
            Reverse(pa.len().ilog2())
                .cmp(&Reverse(pb.len().ilog2()))
                .then_with(|| a.cmp(b))
        });
        entries.insert(0, (CORPUS_TERM.to_string(), lengths));

        let bounds = entries
            .iter()
            .enumerate()
            .map(|(idx, (_, postings))| {
                if idx == 0 {
                    0
                } else {
                    corpus.bound(postings.iter().copied())
                }
            })
            .collect();
        let postings = posting_lists(entries.iter().map(|(_, p)| p.as_slice()), &mut ctx)?;
        Ok(Some((
            vec![InvertedEntries {
                terms: entries.into_iter().map(|(term, _)| term).collect(),
                postings,
                bounds,
            }],
            (),
        )))
    }

    fn buffered_bytes(&self) -> u64 {
        self.terms
            .iter()
            .map(|(term, rows)| (term.len() + rows.len() * 8) as u64)
            .sum()
    }
}
