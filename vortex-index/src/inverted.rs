// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A full-text inverted index: for each token, the rows holding it and how often.
//!
//! Answers [`text_match`](crate::text_match::text_match) exactly, since it tokenizes the same way,
//! and ranks rows for a free-text query with BM25 through [`IndexRows::search_bm25`].
//!
//! One extra entry under the empty term, which no token can equal, lists every non-null row with
//! its token count. BM25 needs those document lengths, and keeping them as an ordinary entry
//! means every index row stays self-contained and a search reads only the entries it scores.

use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::or_collect;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::validity::Validity;
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
use crate::postings::u32s_from_varints;
use crate::postings::u32s_to_varints;
use crate::rows::Hit;
use crate::rows::IndexRows;
use crate::rows::top_k;
use crate::text_match::TextMatch;
use crate::tokenize::tokenize;

/// Stable registry id of the full-text index.
pub const INVERTED_INDEX_ID: &str = "vortex.idx.inverted";

const TERM_FIELD: &str = "term";
const POSTINGS_FIELD: &str = "postings";
const FREQS_FIELD: &str = "freqs";

/// The entry holding every non-null row and its token count.
const DOC_LENGTHS_TERM: &str = "";

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

/// Sorted terms, each with its rows and the term's frequency in each row, in row order.
///
/// Frequencies are stored as varints, so a typical posting costs its roaring bits plus one byte.
#[derive(Debug)]
pub struct InvertedEntries {
    terms: Vec<String>,
    postings: Vec<RoaringBitmap>,
    freqs: Vec<Vec<u32>>,
}

impl InvertedEntries {
    fn get(&self, term: &str) -> Option<(&RoaringBitmap, &[u32])> {
        self.terms
            .binary_search_by(|probe| probe.as_str().cmp(term))
            .ok()
            .map(|idx| (&self.postings[idx], self.freqs[idx].as_slice()))
    }
}

fn find<'a>(
    chunks: &'a [Arc<InvertedEntries>],
    term: &str,
) -> Option<(&'a RoaringBitmap, &'a [u32])> {
    chunks.iter().find_map(|chunk| chunk.get(term))
}

fn index_fields() -> StructFields {
    let names: FieldNames = vec![TERM_FIELD, POSTINGS_FIELD, FREQS_FIELD].into();
    StructFields::new(
        names,
        vec![
            DType::Utf8(NonNullable),
            DType::Binary(NonNullable),
            DType::Binary(NonNullable),
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
        let freqs = chunk.freqs.iter().map(|f| u32s_to_varints(f));
        Ok(StructArray::try_new_with_dtype(
            vec![
                VarBinViewArray::from_iter_str(&chunk.terms).into_array(),
                VarBinViewArray::from_iter_bin(postings).into_array(),
                VarBinViewArray::from_iter_bin(freqs).into_array(),
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
        let freqs = rows
            .binary_field(FREQS_FIELD, ctx)?
            .iter()
            .map(|bytes| u32s_from_varints(bytes.as_slice()))
            .collect::<VortexResult<Vec<_>>>()?;
        for (bitmap, freqs) in postings.iter().zip(&freqs) {
            vortex_ensure_eq!(
                bitmap.len(),
                freqs.len() as u64,
                "Inverted index entry has mismatched postings and frequencies"
            );
        }
        // Rows are sorted by term, and the chunk boundaries never split that order.
        debug_assert!(terms.is_sorted());
        debug_assert_eq!(rows.len(), terms.len());
        Ok(InvertedEntries {
            terms,
            postings,
            freqs,
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
            let Some((rows, _)) = find(chunks, term) else {
                return Ok(RowLocator::empty_rows());
            };
            result = Some(match result {
                None => rows.clone(),
                Some(acc) => acc & rows,
            });
        }
        Ok(RowLocator::Rows(result.unwrap_or_default()))
    }
}

impl IndexRows<InvertedIndex> {
    /// The `k` rows that score best for `query` under BM25, best first.
    ///
    /// A row scores if it holds any query token. Only the entries for the query's tokens and the
    /// document lengths are read.
    pub async fn search_bm25(&self, query: &str, k: usize) -> VortexResult<Vec<Hit>> {
        let terms = query_terms(query);
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let filter = terms_filter(terms.iter().cloned().chain([DOC_LENGTHS_TERM.to_string()]));
        let chunks = self.chunks(filter).await?;

        let Some((docs, doc_lengths)) = find(&chunks, DOC_LENGTHS_TERM) else {
            return Ok(Vec::new());
        };
        let doc_count = docs.len() as f32;
        if doc_count == 0.0 {
            return Ok(Vec::new());
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "an average length fits an f32 with ample precision for scoring"
        )]
        let avg_length =
            (doc_lengths.iter().map(|&l| f64::from(l)).sum::<f64>() / f64::from(doc_count)) as f32;

        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in &terms {
            let Some((rows, freqs)) = find(&chunks, term) else {
                continue;
            };
            let df = rows.len() as f32;
            let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln();
            for (row, &tf) in rows.iter().zip(freqs) {
                // Every row holding a token is non-null, so it has a document length.
                let rank = docs.rank(row);
                let length = rank
                    .checked_sub(1)
                    .and_then(|idx| doc_lengths.get(usize::try_from(idx).ok()?))
                    .copied()
                    .ok_or_else(|| vortex_err!("Row {row} has no document length"))?;
                let tf = tf as f32;
                let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length as f32 / avg_length);
                *scores.entry(row).or_default() += idf * tf * (BM25_K1 + 1.0) / (tf + norm);
            }
        }

        let hits = scores
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
    /// them sorted.
    terms: HashMap<String, Vec<(u32, u32)>>,
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
                self.terms.entry(term).or_default().push((row, tf));
            }
            self.terms
                .entry(DOC_LENGTHS_TERM.to_string())
                .or_default()
                .push((row, length));
        }
        Ok(())
    }

    fn finish(self) -> VortexResult<Option<(Vec<InvertedEntries>, ())>> {
        let mut entries: Vec<_> = self.terms.into_iter().collect();
        entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        let mut terms = Vec::with_capacity(entries.len());
        let mut postings = Vec::with_capacity(entries.len());
        let mut freqs = Vec::with_capacity(entries.len());
        for (term, rows) in entries {
            terms.push(term);
            postings.push(rows.iter().map(|(row, _)| *row).collect::<RoaringBitmap>());
            freqs.push(rows.into_iter().map(|(_, tf)| tf).collect());
        }
        Ok(Some((
            vec![InvertedEntries {
                terms,
                postings,
                freqs,
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
