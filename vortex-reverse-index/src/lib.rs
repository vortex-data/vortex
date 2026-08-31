//! A simple `value -> rows` reverse index over any column, demonstrating [`vortex_layout`]'s
//! `vortex.indexed` layout [`IndexVTable`] contract with a minimal concrete index kind.
//!
//! This is deliberately narrow: it supports only equality (`column == literal`), answered exactly
//! via a sorted `key -> postings` table, the same shape as the indexed layout's own test-only
//! `exact_value` index but generalized to any dtype instead of being fixed to `Utf8`. It exists as
//! a worked, non-test example of a concrete index kind for the `vortex.indexed` layout prototype
//! described in [vortex-data/vortex#9024](https://github.com/vortex-data/vortex/issues/9024).

use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::layouts::indexed::IndexBuilder;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexId;
use vortex_layout::layouts::indexed::IndexQueryPlan;
use vortex_layout::layouts::indexed::IndexResolve;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::IndexVTableRef;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_map::HashMap;

#[cfg(test)]
mod tests;

/// Stable registry id of this index kind.
pub const REVERSE_INDEX_ID: &str = "vortex.idx.reverse_index";

const KEY_FIELD: &str = "key";
const POSTINGS_FIELD: &str = "postings";

/// A value -> rows reverse index over a single column of any dtype.
///
/// One row per distinct value, sorted by key, with a roaring posting list of the rows holding it.
/// Sorting the key column gives it a useful zone map, so probing the index is a pruned scan rather
/// than a full decode. Equality is answered exactly: [`IndexVTable::plan`] only claims
/// `column == literal`, so the probe's mask is the answer, not just a filter to re-check.
#[derive(Debug)]
pub struct ReverseIndex;

impl ReverseIndex {
    /// A shared handle to this index kind, ready to register into an
    /// [`IndexSession`](vortex_layout::layouts::indexed::IndexSession).
    pub fn new_ref() -> IndexVTableRef {
        Arc::new(Self)
    }
}

impl IndexVTable for ReverseIndex {
    fn id(&self) -> IndexId {
        static ID: CachedId = CachedId::new(REVERSE_INDEX_ID);
        *ID
    }

    fn supports_dtype(&self, dtype: &DType) -> bool {
        // Every other dtype decodes to a `Scalar` and has a canonical `ArrayBuilder`; `Union` and
        // `Variant` do not yet, so decline rather than panic building their key column.
        !matches!(dtype, DType::Union(..) | DType::Variant(_))
    }

    fn builder(
        &self,
        dtype: &DType,
        _options: &[u8],
        _data_block_len: Option<u64>,
        _session: &VortexSession,
    ) -> VortexResult<Box<dyn IndexBuilder>> {
        Ok(Box::new(Builder {
            dtype: dtype.clone(),
            postings: HashMap::new(),
        }))
    }

    fn plan(
        &self,
        expr: &BoundExpression,
        dtype: &DType,
        _options: &[u8],
    ) -> VortexResult<Option<IndexQueryPlan>> {
        // Only `<column> == <literal>`.
        if !expr.is::<Binary>() || *expr.as_::<Binary>() != Operator::Eq {
            return Ok(None);
        }
        if !expr.child(0).is_root() || !expr.child(1).is::<Literal>() {
            return Ok(None);
        }

        let target = expr.child(1).as_::<Literal>();
        if !target.dtype().eq_ignore_nullability(dtype) {
            // Binding rejects mismatched dtypes for ordinary comparisons, but exempts `Extension`
            // dtypes from that check; decline rather than guess at cross-type equality.
            return Ok(None);
        }
        if target.is_null() {
            // Null literal: `column == NULL` never matches under normal equality, and this index
            // does not model that, so decline rather than claim it incorrectly.
            return Ok(None);
        }
        let target = target.clone();

        Ok(Some(IndexQueryPlan {
            exactness: IndexExactness::Exact,
            filter: eq(col(KEY_FIELD), lit(target.clone())),
            resolve: Arc::new(Resolve { target }),
        }))
    }
}

fn index_fields(key_dtype: DType) -> StructFields {
    let names: FieldNames = vec![KEY_FIELD, POSTINGS_FIELD].into();
    StructFields::new(names, vec![key_dtype, DType::Binary(NonNullable)])
}

struct Builder {
    dtype: DType,
    /// Deduplicated by scalar equality; sorted into key order in `finish`, which is what gives
    /// the key column a useful zone map.
    postings: HashMap<Scalar, RoaringBitmap>,
}

impl IndexBuilder for Builder {
    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for idx in 0..chunk.len() {
            let value = chunk.execute_scalar(idx, ctx)?;
            if value.is_null() {
                continue;
            }
            self.postings
                .entry(value)
                .or_default()
                .insert(u32::try_from(row_offset + idx as u64)?);
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<Option<(SendableArrayStream, Vec<u8>)>> {
        let Builder { dtype, postings } = *self;

        let mut entries: Vec<(Scalar, RoaringBitmap)> = postings.into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| {
            a.partial_cmp(b)
                .vortex_expect("keys were all decoded from the same column, so they share a dtype")
        });

        let key_dtype = dtype.as_nonnullable();
        let mut key_builder = builder_with_capacity(&key_dtype, entries.len());
        let mut lists = Vec::with_capacity(entries.len());
        for (key, bitmap) in &entries {
            // Keys are never null (`push` skips them), but may carry the source column's nullable
            // dtype; the key column itself is non-nullable, so normalize before appending.
            key_builder.append_scalar(&key.cast(&key_dtype)?)?;
            let mut buffer = Vec::with_capacity(bitmap.serialized_size());
            bitmap
                .serialize_into(&mut buffer)
                .map_err(|err| vortex_err!("Failed to serialize postings: {err}"))?;
            lists.push(buffer);
        }

        let len = entries.len();
        let array = StructArray::try_new_with_dtype(
            vec![
                key_builder.finish(),
                VarBinViewArray::from_iter_bin(lists).into_array(),
            ],
            index_fields(key_dtype),
            len,
            Validity::NonNullable,
        )?;

        Ok(Some((array.into_array().to_array_stream().boxed(), vec![])))
    }

    fn buffered_bytes(&self) -> u64 {
        self.postings
            .values()
            .map(|bitmap| bitmap.serialized_size() as u64)
            .sum()
    }
}

struct Resolve {
    target: Scalar,
}

impl IndexResolve for Resolve {
    fn resolve(
        &self,
        postings: &ArrayRef,
        _data_row_count: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RowLocator> {
        let entries = postings.clone().execute::<StructArray>(ctx)?;
        let keys = entries.unmasked_field_by_name(KEY_FIELD)?;
        let lists = entries
            .unmasked_field_by_name(POSTINGS_FIELD)?
            .clone()
            .execute::<VarBinViewArray>(ctx)?;

        let Some(idx) = keys
            .search_sorted(&self.target, SearchSortedSide::Left)?
            .to_found()
        else {
            return Ok(RowLocator::empty_rows());
        };

        let bitmap = RoaringBitmap::deserialize_from(lists.bytes_at(idx).as_slice())
            .map_err(|err| vortex_err!("Failed to deserialize postings: {err}"))?;
        Ok(RowLocator::Rows(bitmap))
    }
}
