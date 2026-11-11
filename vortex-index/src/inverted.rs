// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A full-text inverted index: for each token, the rows holding it, each with its BM25 score.
//!
//! Answers [`text_match`](crate::text_match::text_match) exactly, since it tokenizes the same way,
//! and ranks rows for a free-text query with BM25 through [`IndexRows::search_bm25`].
//!
//! The index is built once over all the rows it covers, so each posting's BM25 contribution is
//! known at build time and stored in place of its term frequency: a row's score for a query is
//! the sum of its postings' stored scores. Each term also stores the largest of its scores, which
//! bounds what the term can add to any row, so a search skips scoring the rows of terms too
//! common to change its top `k` (MaxScore).

use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
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
use vortex_buffer::ByteBuffer;
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
use crate::postings::bitmap_from_bytes;
use crate::postings::bitmap_to_bytes;
use crate::postings::f32s_to_bytes;
use crate::rows::Hit;
use crate::rows::IndexRows;
use crate::rows::top_k;
use crate::text_match::TextMatch;
use crate::tokenize::tokenize;

/// Stable registry id of the full-text index.
pub const INVERTED_INDEX_ID: &str = "vortex.idx.inverted";

const TERM_FIELD: &str = "term";
const POSTINGS_FIELD: &str = "postings";
const SCORES_FIELD: &str = "scores";
const MAX_SCORE_FIELD: &str = "max_score";

/// BM25 term-frequency saturation.
const BM25_K1: f32 = 1.2;
/// BM25 document-length normalization.
const BM25_B: f32 = 0.75;

/// A full-text inverted index over a UTF-8 column.
#[derive(Clone, Debug)]
pub struct InvertedIndex;

impl InvertedIndex {
    /// A shared handle to this index kind, for registration and write configuration.
    pub fn new_ref() -> IndexVTableRef {
        IndexVTableRef::new(Self)
    }
}

/// Sorted terms, each with its rows and, per row in row order, the term's BM25 score there.
///
/// Scores are little-endian `f32`s, read in place, so a search reads one term's score for one
/// row without decoding the rest.
#[derive(Debug)]
pub struct InvertedEntries {
    terms: Vec<String>,
    postings: Vec<RoaringBitmap>,
    scores: Vec<ByteBuffer>,
    max_scores: Vec<f32>,
}

/// One term's postings.
struct Posting<'a> {
    rows: &'a RoaringBitmap,
    scores: &'a [u8],
    max_score: f32,
}

impl Posting<'_> {
    /// Each row and its score, in row order.
    fn scored(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.rows.iter().zip(
            self.scores
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c)),
        )
    }

    /// The term's score in `row`, if the row holds the term.
    fn score_of(&self, row: u32) -> Option<f32> {
        if !self.rows.contains(row) {
            return None;
        }
        let idx = usize::try_from(self.rows.rank(row) - 1).ok()? * 4;
        let bytes = self.scores.get(idx..idx + 4)?;
        Some(f32::from_le_bytes(bytes.try_into().ok()?))
    }
}

impl InvertedEntries {
    fn get(&self, term: &str) -> Option<Posting<'_>> {
        self.terms
            .binary_search_by(|probe| probe.as_str().cmp(term))
            .ok()
            .map(|idx| Posting {
                rows: &self.postings[idx],
                scores: self.scores[idx].as_slice(),
                max_score: self.max_scores[idx],
            })
    }
}

fn find<'a>(chunks: &'a [Arc<InvertedEntries>], term: &str) -> Option<Posting<'a>> {
    chunks.iter().find_map(|chunk| chunk.get(term))
}

fn index_fields() -> StructFields {
    let names: FieldNames = vec![TERM_FIELD, POSTINGS_FIELD, SCORES_FIELD, MAX_SCORE_FIELD].into();
    StructFields::new(
        names,
        vec![
            DType::Utf8(NonNullable),
            DType::Binary(NonNullable),
            DType::Binary(NonNullable),
            DType::Primitive(PType::F32, NonNullable),
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
        _session: &VortexSession,
    ) -> VortexResult<InvertedBuilder> {
        Ok(InvertedBuilder {
            terms: HashMap::new(),
            docs: 0,
            total_length: 0,
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
        let postings = chunk
            .postings
            .iter()
            .map(bitmap_to_bytes)
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new_with_dtype(
            vec![
                VarBinViewArray::from_iter_str(&chunk.terms).into_array(),
                VarBinViewArray::from_iter_bin(postings).into_array(),
                VarBinViewArray::from_iter_bin(chunk.scores.iter().map(ByteBuffer::as_slice))
                    .into_array(),
                chunk
                    .max_scores
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
        let postings = rows
            .binary_field(POSTINGS_FIELD, ctx)?
            .iter()
            .map(|bytes| bitmap_from_bytes(bytes.as_slice()))
            .collect::<VortexResult<Vec<_>>>()?;
        let scores = rows.binary_field(SCORES_FIELD, ctx)?;
        let max_scores = rows
            .field(MAX_SCORE_FIELD)?
            .execute::<PrimitiveArray>(ctx)?
            .as_slice::<f32>()
            .to_vec();
        for (bitmap, scores) in postings.iter().zip(&scores) {
            vortex_ensure_eq!(
                bitmap.len() * 4,
                scores.len() as u64,
                "Inverted index entry has mismatched postings and scores"
            );
        }
        // Rows are sorted by term, and the chunk boundaries never split that order.
        debug_assert!(terms.is_sorted());
        debug_assert_eq!(rows.len(), terms.len());
        Ok(InvertedEntries {
            terms,
            postings,
            scores,
            max_scores,
        })
    }

    fn resolve(
        &self,
        terms: &Vec<String>,
        chunks: &[Arc<InvertedEntries>],
        _data_row_count: u64,
        _options: &(),
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<RowLocator> {
        let mut result: Option<RoaringBitmap> = None;
        for term in terms {
            let Some(posting) = find(chunks, term) else {
                return Ok(RowLocator::empty_rows());
            };
            result = Some(match result {
                None => posting.rows.clone(),
                Some(acc) => acc & posting.rows,
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
        let chunks = self.chunks_for_keys(&terms).await?;
        let mut lists: Vec<Posting<'_>> = terms
            .iter()
            .filter_map(|term| find(&chunks, term))
            .collect();
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
            scored = merge(scored, lists[walked].scored());
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
                    if let Some(extra) = list.score_of(*row) {
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
    /// For each term, `(row, frequency, row length)` in row order. Rows arrive in order, so
    /// appending keeps them sorted.
    terms: HashMap<String, Vec<(u32, u32, u32)>>,
    docs: u64,
    total_length: u64,
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
            for (term, tf) in row_terms.drain() {
                self.terms.entry(term).or_default().push((row, tf, length));
            }
            self.docs += 1;
            self.total_length += u64::from(length);
        }
        Ok(())
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "corpus counts only weight scores, so f32 precision is ample"
    )]
    fn finish(self) -> VortexResult<Option<(Vec<InvertedEntries>, ())>> {
        let doc_count = self.docs as f32;
        let avg_length = if self.docs == 0 {
            1.0
        } else {
            (self.total_length as f64 / self.docs as f64) as f32
        };
        // The per-row part of the BM25 denominator: k1 * (1 - b + b * len / avg).
        let norm_a = BM25_K1 * (1.0 - BM25_B);
        let norm_b = BM25_K1 * BM25_B / avg_length;

        let mut entries: Vec<_> = self.terms.into_iter().collect();
        entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        let mut terms = Vec::with_capacity(entries.len());
        let mut postings = Vec::with_capacity(entries.len());
        let mut scores = Vec::with_capacity(entries.len());
        let mut max_scores = Vec::with_capacity(entries.len());
        for (term, rows) in entries {
            let df = rows.len() as f32;
            let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln();
            let weight = idf * (BM25_K1 + 1.0);
            let term_scores: Vec<f32> = rows
                .iter()
                .map(|&(_, tf, length)| {
                    let tf = tf as f32;
                    weight * tf / (tf + norm_a + norm_b * length as f32)
                })
                .collect();
            terms.push(term);
            postings.push(rows.iter().map(|(row, ..)| *row).collect::<RoaringBitmap>());
            max_scores.push(term_scores.iter().copied().fold(0.0, f32::max));
            scores.push(ByteBuffer::from(f32s_to_bytes(&term_scores)));
        }
        Ok(Some((
            vec![InvertedEntries {
                terms,
                postings,
                scores,
                max_scores,
            }],
            (),
        )))
    }

    fn buffered_bytes(&self) -> u64 {
        self.terms
            .iter()
            .map(|(term, rows)| (term.len() + rows.len() * 12) as u64)
            .sum()
    }
}
