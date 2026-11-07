// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A trigram skip index: which blocks of rows contain each byte trigram.
//!
//! Serves `LIKE` and `ILIKE` with a literal pattern as a superset: a block that lacks any trigram
//! of the pattern's literal runs cannot hold a match, so it is pruned, and the real predicate
//! rechecks the surviving blocks.

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
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::or_collect;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
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
use crate::tokenize::like_literal_runs;
use crate::tokenize::trigrams;

/// Stable registry id of the trigram index.
pub const TRIGRAM_INDEX_ID: &str = "vortex.idx.trigram";

/// Block length used when neither the options nor the data child's chunking pick one.
pub const DEFAULT_TRIGRAM_BLOCK_LEN: u64 = 1024;

const TRIGRAM_FIELD: &str = "trigram";
const BLOCKS_FIELD: &str = "blocks";

/// A trigram skip index over a UTF-8 column.
#[derive(Clone, Debug)]
pub struct TrigramIndex;

impl TrigramIndex {
    /// A shared handle to this index kind, for registration and write configuration.
    pub fn new_ref() -> IndexVTableRef {
        IndexVTableRef::new(Self)
    }
}

/// Options of the trigram index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrigramOptions {
    /// Rows per block. Zero picks the data child's block length, or
    /// [`DEFAULT_TRIGRAM_BLOCK_LEN`]. A built index always records the length it used.
    pub block_len: u64,
}

/// Sorted trigrams and the blocks holding each.
#[derive(Debug)]
pub struct TrigramEntries {
    trigrams: Vec<u32>,
    blocks: Vec<RoaringBitmap>,
}

fn index_fields() -> StructFields {
    let names: FieldNames = vec![TRIGRAM_FIELD, BLOCKS_FIELD].into();
    StructFields::new(
        names,
        vec![
            DType::Primitive(PType::U32, NonNullable),
            DType::Binary(NonNullable),
        ],
    )
}

impl IndexVTable for TrigramIndex {
    type Options = TrigramOptions;
    type Builder = TrigramBuilder;
    /// The distinct trigrams every match must contain, sorted.
    type Query = Vec<u32>;
    type Chunk = TrigramEntries;

    fn id(&self) -> IndexId {
        static ID: CachedId = CachedId::new(TRIGRAM_INDEX_ID);
        *ID
    }

    fn serialize_options(&self, options: &TrigramOptions) -> Vec<u8> {
        options.block_len.to_le_bytes().to_vec()
    }

    fn deserialize_options(&self, options: &[u8]) -> VortexResult<TrigramOptions> {
        if options.is_empty() {
            return Ok(TrigramOptions::default());
        }
        let bytes: [u8; 8] = options
            .try_into()
            .map_err(|_| vortex_error::vortex_err!("Trigram options must be 8 bytes"))?;
        Ok(TrigramOptions {
            block_len: u64::from_le_bytes(bytes),
        })
    }

    fn index_dtype(&self, dtype: &DType, _options: &TrigramOptions) -> Option<DType> {
        dtype
            .is_utf8()
            .then(|| DType::Struct(index_fields(), NonNullable))
    }

    fn builder(
        &self,
        _dtype: &DType,
        options: &TrigramOptions,
        data_block_len: Option<u64>,
        _session: &VortexSession,
    ) -> VortexResult<TrigramBuilder> {
        let block_len = match options.block_len {
            0 => data_block_len.unwrap_or(DEFAULT_TRIGRAM_BLOCK_LEN),
            len => len,
        };
        vortex_ensure!(block_len > 0, "Trigram block length must be positive");
        Ok(TrigramBuilder {
            block_len,
            blocks: HashMap::new(),
        })
    }

    fn plan(
        &self,
        expr: &BoundExpression,
        _dtype: &DType,
        index_dtype: &DType,
        _options: &TrigramOptions,
    ) -> VortexResult<Option<IndexQueryPlan<Vec<u32>>>> {
        let Some(like) = expr.as_opt::<Like>() else {
            return Ok(None);
        };
        if like.negated || !expr.child(0).is_root() || !expr.child(1).is::<Literal>() {
            return Ok(None);
        }
        let Some(pattern) = expr.child(1).as_::<Literal>().as_utf8_opt() else {
            return Ok(None);
        };
        let Some(pattern) = pattern.value() else {
            return Ok(None);
        };
        // The index lowercases ASCII only, so it cannot vouch for Unicode case folding.
        if like.case_insensitive && !pattern.as_str().is_ascii() {
            return Ok(None);
        }
        let Some(runs) = like_literal_runs(pattern.as_str()) else {
            return Ok(None);
        };
        let mut wanted: Vec<u32> = runs
            .iter()
            .flat_map(|run| trigrams(run.as_bytes()))
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        let Some(filter) = or_collect(
            wanted
                .iter()
                .map(|&trigram| eq(col(TRIGRAM_FIELD), lit(trigram))),
        ) else {
            // No literal run is long enough to hold a trigram.
            return Ok(None);
        };

        Ok(Some(IndexQueryPlan {
            exactness: IndexExactness::Superset,
            filter: filter.bind(index_dtype)?,
            query: wanted,
        }))
    }

    fn encode(
        &self,
        chunk: &TrigramEntries,
        _options: &TrigramOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let blocks = chunk
            .blocks
            .iter()
            .map(bitmap_to_bytes)
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new_with_dtype(
            vec![
                chunk
                    .trigrams
                    .iter()
                    .copied()
                    .collect::<PrimitiveArray>()
                    .into_array(),
                VarBinViewArray::from_iter_bin(blocks).into_array(),
            ],
            index_fields(),
            chunk.trigrams.len(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn decode(
        &self,
        chunk: ArrayRef,
        _options: &TrigramOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<TrigramEntries> {
        let rows = IndexRowsChunk::new(chunk, ctx)?;
        let trigrams = rows
            .field(TRIGRAM_FIELD)?
            .execute::<PrimitiveArray>(ctx)?
            .as_slice::<u32>()
            .to_vec();
        let blocks = rows
            .binary_field(BLOCKS_FIELD, ctx)?
            .iter()
            .map(|bytes| bitmap_from_bytes(bytes.as_slice()))
            .collect::<VortexResult<_>>()?;
        Ok(TrigramEntries { trigrams, blocks })
    }

    fn resolve(
        &self,
        wanted: &Vec<u32>,
        chunks: &[Arc<TrigramEntries>],
        _data_row_count: u64,
        options: &TrigramOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<RowLocator> {
        let block_len = options.block_len.max(1);
        let mut result: Option<RoaringBitmap> = None;
        for trigram in wanted {
            let blocks = chunks.iter().find_map(|chunk| {
                chunk
                    .trigrams
                    .binary_search(trigram)
                    .ok()
                    .map(|idx| &chunk.blocks[idx])
            });
            let Some(blocks) = blocks else {
                // A trigram no row holds: nothing can match.
                return Ok(RowLocator::Blocks {
                    block_len,
                    ids: RoaringBitmap::new(),
                });
            };
            result = Some(match result {
                None => blocks.clone(),
                Some(acc) => acc & blocks,
            });
            if result.as_ref().is_some_and(RoaringBitmap::is_empty) {
                break;
            }
        }
        Ok(RowLocator::Blocks {
            block_len,
            ids: result.unwrap_or_default(),
        })
    }
}

/// Builds a [`TrigramIndex`] while the column streams past.
pub struct TrigramBuilder {
    block_len: u64,
    blocks: HashMap<u32, RoaringBitmap>,
}

impl IndexBuilder for TrigramBuilder {
    type Options = TrigramOptions;
    type Chunk = TrigramEntries;

    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let values = chunk.clone().execute::<VarBinViewArray>(ctx)?;
        let valid = values.validity()?.execute_mask(values.len(), ctx)?;
        for idx in 0..values.len() {
            if !valid.value(idx) {
                continue;
            }
            let block = u32::try_from((row_offset + idx as u64) / self.block_len)?;
            let bytes = values.bytes_at(idx);
            for trigram in trigrams(bytes.as_slice()) {
                self.blocks.entry(trigram).or_default().insert(block);
            }
        }
        Ok(())
    }

    fn finish(self) -> VortexResult<Option<(Vec<TrigramEntries>, TrigramOptions)>> {
        let mut entries: Vec<_> = self.blocks.into_iter().collect();
        entries.sort_unstable_by_key(|(trigram, _)| *trigram);
        let (trigrams, blocks) = entries.into_iter().unzip();
        Ok(Some((
            vec![TrigramEntries { trigrams, blocks }],
            TrigramOptions {
                block_len: self.block_len,
            },
        )))
    }

    fn buffered_bytes(&self) -> u64 {
        self.blocks
            .values()
            .map(|bitmap| bitmap.serialized_size() as u64 + 4)
            .sum()
    }
}
