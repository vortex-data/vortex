use std::sync::{Arc, RwLock};

use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_panic, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::EncodingRef;
use crate::stats::{Stat, Statistics, StatsSet};
use crate::{ArrayDType, ArrayData, ArrayMetadata};

/// Owned [`ArrayData`] with serialized metadata, backed by heap-allocated memory.
#[derive(Clone, Debug)]
pub(super) struct OwnedArrayData {
    pub(super) encoding: EncodingRef,
    pub(super) dtype: DType, // FIXME(ngates): Arc?
    pub(super) len: usize,
    pub(super) metadata: Arc<dyn ArrayMetadata>,
    pub(super) buffer: Option<Buffer>,
    pub(super) children: Arc<[ArrayData]>,
    pub(super) stats_set: Arc<RwLock<StatsSet>>,
}

impl OwnedArrayData {
    pub fn encoding(&self) -> EncodingRef {
        self.encoding
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn metadata(&self) -> &Arc<dyn ArrayMetadata> {
        &self.metadata
    }

    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    pub fn into_buffer(self) -> Option<Buffer> {
        self.buffer
    }

    // We want to allow these panics because they are indicative of implementation error.
    #[allow(clippy::panic_in_result_fn)]
    pub fn child(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<&ArrayData> {
        match self.children.get(index) {
            None => vortex_bail!(
                "ArrayData::child({}): child {index} not found",
                self.encoding.id().as_ref()
            ),
            Some(child) => {
                assert_eq!(
                    child.dtype(),
                    dtype,
                    "child {index} requested with incorrect dtype for encoding {}",
                    self.encoding().id().as_ref(),
                );
                assert_eq!(
                    child.len(),
                    len,
                    "child {index} requested with incorrect length for encoding {}",
                    self.encoding.id().as_ref(),
                );
                Ok(child)
            }
        }
    }

    pub fn nchildren(&self) -> usize {
        self.children.len()
    }

    pub fn children(&self) -> &[ArrayData] {
        &self.children
    }

    pub fn statistics(&self) -> &dyn Statistics {
        self
    }
}

impl Statistics for OwnedArrayData {
    fn get(&self, stat: Stat) -> Option<Scalar> {
        self.stats_set
            .read()
            .unwrap_or_else(|_| {
                vortex_panic!(
                    "Failed to acquire read lock on stats map while getting {}",
                    stat
                )
            })
            .get(stat)
            .cloned()
    }

    fn to_set(&self) -> StatsSet {
        self.stats_set
            .read()
            .unwrap_or_else(|_| vortex_panic!("Failed to acquire read lock on stats map"))
            .clone()
    }

    fn set(&self, stat: Stat, value: Scalar) {
        self.stats_set
            .write()
            .unwrap_or_else(|_| {
                vortex_panic!(
                    "Failed to acquire write lock on stats map while setting {} to {}",
                    stat,
                    value
                )
            })
            .set(stat, value);
    }

    fn clear(&self, stat: Stat) {
        self.stats_set
            .write()
            .unwrap_or_else(|_| vortex_panic!("Failed to acquire write lock on stats map"))
            .clear(stat);
    }

    fn compute(&self, stat: Stat) -> Option<Scalar> {
        if let Some(s) = self.get(stat) {
            return Some(s);
        }

        let computed = self
            .encoding()
            .compute_statistics(&ArrayData::from(self.clone()), stat)
            .ok()?;

        self.stats_set
            .write()
            .unwrap_or_else(|_| {
                vortex_panic!("Failed to write to stats map while computing {}", stat)
            })
            .extend(computed);
        self.get(stat)
    }

    fn retain_only(&self, stats: &[Stat]) {
        self.stats_set
            .write()
            .unwrap_or_else(|_| vortex_panic!("Failed to acquire write lock on stats map"))
            .retain_only(stats);
    }
}
