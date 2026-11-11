// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A cached copy of an index's sorted key column, so a lookup finds its keys' rows by binary
//! search and reads just those rows, rather than filtering the whole index through its zone maps.

use std::any::Any;
use std::sync::Arc;

use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::expr::col;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::layouts::indexed::IndexVTable;

use crate::rows::IndexRows;
use crate::rows::IndexStep;

/// A key type an index sorts its rows by: the first field of its rows.
pub(crate) trait IndexKey: Ord + Send + Sync + 'static {
    /// Every key of the index, in row order.
    type Column: KeyColumn<Self> + Send + Sync + 'static;
}

/// A column of sorted keys.
pub(crate) trait KeyColumn<K: ?Sized> {
    /// Read the column from the executed key arrays of an index, in row order.
    fn load(
        arrays: Vec<vortex_array::ArrayRef>,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<Self>
    where
        Self: Sized;

    /// The row holding `key`, if any.
    fn position(&self, key: &K) -> Option<u64>;
}

impl IndexKey for u32 {
    type Column = Vec<u32>;
}

impl KeyColumn<u32> for Vec<u32> {
    fn load(
        arrays: Vec<vortex_array::ArrayRef>,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<Self> {
        let mut keys = Vec::new();
        for array in arrays {
            keys.extend_from_slice(array.execute::<PrimitiveArray>(ctx)?.as_slice::<u32>());
        }
        Ok(keys)
    }

    fn position(&self, key: &u32) -> Option<u64> {
        self.binary_search(key).ok().map(|idx| idx as u64)
    }
}

/// Sorted strings, packed into one buffer.
pub(crate) struct Strings {
    bytes: Vec<u8>,
    /// Start of each string in `bytes`, plus the end of the last.
    offsets: Vec<usize>,
}

impl Strings {
    fn get(&self, idx: usize) -> &[u8] {
        &self.bytes[self.offsets[idx]..self.offsets[idx + 1]]
    }
}

impl IndexKey for String {
    type Column = Strings;
}

impl KeyColumn<String> for Strings {
    fn load(
        arrays: Vec<vortex_array::ArrayRef>,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<Self> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0];
        for array in arrays {
            let values = array.execute::<VarBinViewArray>(ctx)?;
            for idx in 0..values.len() {
                bytes.extend_from_slice(values.bytes_at(idx).as_slice());
                offsets.push(bytes.len());
            }
        }
        Ok(Self { bytes, offsets })
    }

    fn position(&self, key: &String) -> Option<u64> {
        let count = self.offsets.len() - 1;
        let (mut lo, mut hi) = (0, count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.get(mid).cmp(key.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u64),
            }
        }
        None
    }
}

impl<V: IndexVTable> IndexRows<V> {
    /// The index's key column, read on first use and then kept.
    fn key_column_step<K: IndexKey>(self: &Arc<Self>) -> VortexResult<IndexStep<Arc<K::Column>>> {
        if let Some(keys) = self
            .memo
            .get()
            .and_then(|memo| Arc::clone(memo).downcast::<K::Column>().ok())
        {
            return Ok(IndexStep::Done(keys));
        }
        let key_field = self
            .index_dtype
            .as_struct_fields_opt()
            .and_then(|fields| fields.names().iter().next().cloned())
            .ok_or_else(|| vortex_err!("Index rows have no key field"))?;
        let projection = col(key_field).bind(&self.index_dtype)?;
        let this = Arc::clone(self);
        Ok(IndexStep::Read(
            Box::new(self.scan().with_projection(projection)),
            Box::new(move |arrays| {
                let mut ctx = this.session.create_execution_ctx();
                let keys = Arc::new(K::Column::load(arrays, &mut ctx)?);
                // A concurrent first lookup may have set it already; both read the same column.
                drop(
                    this.memo
                        .set(Arc::clone(&keys) as Arc<dyn Any + Send + Sync>),
                );
                Ok(IndexStep::Done(keys))
            }),
        ))
    }

    /// Decode the rows holding `keys`, found through the cached key column. Absent keys are
    /// skipped.
    pub(crate) fn chunks_for_keys_step<K: IndexKey>(
        self: &Arc<Self>,
        keys: Vec<K>,
    ) -> VortexResult<IndexStep<Vec<Arc<V::Chunk>>>> {
        let this = Arc::clone(self);
        self.key_column_step::<K>()?.and_then(move |column| {
            let mut rows: Vec<u64> = keys.iter().filter_map(|key| column.position(key)).collect();
            if rows.is_empty() {
                return Ok(IndexStep::Done(Vec::new()));
            }
            rows.sort_unstable();
            rows.dedup();
            this.chunks_at_step(rows)
        })
    }
}
