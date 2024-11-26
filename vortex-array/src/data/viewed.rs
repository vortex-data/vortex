use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use enum_iterator::all;
use itertools::Itertools;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_err, VortexExpect as _, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::encoding::opaque::OpaqueEncoding;
use crate::encoding::EncodingRef;
use crate::stats::{Stat, Statistics, StatsSet};
use crate::visitor::ArrayVisitor;
use crate::{flatbuffers as fb, ArrayData, ArrayMetadata, Context};

/// Zero-copy view over flatbuffer-encoded array data, created without eager serialization.
#[derive(Clone)]
pub(super) struct ViewedArrayData {
    pub(super) encoding: EncodingRef,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) metadata: Arc<dyn ArrayMetadata>,
    pub(super) flatbuffer: Buffer,
    pub(super) flatbuffer_loc: usize,
    pub(super) buffers: Arc<[Buffer]>,
    pub(super) ctx: Arc<Context>,
}

impl Debug for ViewedArrayData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayView")
            .field("encoding", &self.encoding)
            .field("dtype", &self.dtype)
            .field("buffers", &self.buffers)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl ViewedArrayData {
    pub fn flatbuffer(&self) -> fb::Array {
        unsafe {
            let tab = flatbuffers::Table::new(self.flatbuffer.as_ref(), self.flatbuffer_loc);
            fb::Array::init_from_table(tab)
        }
    }

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

    pub fn metadata_bytes(&self) -> Option<&[u8]> {
        self.flatbuffer().metadata().map(|m| m.bytes())
    }

    // TODO(ngates): should we separate self and DType lifetimes? Should DType be cloned?
    pub fn child(&self, idx: usize, dtype: &DType, len: usize) -> VortexResult<Self> {
        let child = self
            .array_child(idx)
            .ok_or_else(|| vortex_err!("ArrayView: array_child({idx}) not found"))?;
        let flatbuffer_loc = child._tab.loc();

        let encoding = self
            .ctx
            .lookup_encoding(child.encoding())
            .unwrap_or_else(|| {
                // We must return an EncodingRef, which requires a static reference.
                // OpaqueEncoding however must be created dynamically, since we do not know ahead
                // of time which of the ~65,000 unknown code IDs we will end up seeing. Thus, we
                // allocate (and leak) 2 bytes of memory to create a new encoding.
                Box::leak(Box::new(OpaqueEncoding(child.encoding())))
            });

        let metadata = encoding.load_metadata(child.metadata().map(|m| m.bytes()))?;

        Ok(Self {
            encoding,
            dtype: dtype.clone(),
            len,
            metadata,
            flatbuffer: self.flatbuffer.clone(),
            flatbuffer_loc,
            buffers: self.buffers.clone(),
            ctx: self.ctx.clone(),
        })
    }

    fn array_child(&self, idx: usize) -> Option<fb::Array> {
        let children = self.flatbuffer().children()?;
        (idx < children.len()).then(|| children.get(idx))
    }

    pub fn nchildren(&self) -> usize {
        self.flatbuffer().children().map(|c| c.len()).unwrap_or(0)
    }

    pub fn children(&self) -> Vec<ArrayData> {
        let mut collector = ChildrenCollector::default();
        self.encoding()
            .accept(&ArrayData::from(self.clone()), &mut collector)
            .vortex_expect("Failed to get children");
        collector.children
    }

    pub fn buffer(&self) -> Option<&Buffer> {
        self.flatbuffer()
            .buffer_index()
            .map(|idx| &self.buffers[idx as usize])
    }

    pub fn statistics(&self) -> &dyn Statistics {
        self
    }
}

#[derive(Default, Debug)]
struct ChildrenCollector {
    children: Vec<ArrayData>,
}

impl ArrayVisitor for ChildrenCollector {
    fn visit_child(&mut self, _name: &str, array: &ArrayData) -> VortexResult<()> {
        self.children.push(array.clone());
        Ok(())
    }
}

impl Statistics for ViewedArrayData {
    fn get(&self, stat: Stat) -> Option<Scalar> {
        match stat {
            Stat::Max => {
                let max = self.flatbuffer().stats()?.max();
                max.and_then(|v| ScalarValue::try_from(v).ok())
                    .map(|v| Scalar::new(self.dtype.clone(), v))
            }
            Stat::Min => {
                let min = self.flatbuffer().stats()?.min();
                min.and_then(|v| ScalarValue::try_from(v).ok())
                    .map(|v| Scalar::new(self.dtype.clone(), v))
            }
            Stat::IsConstant => self.flatbuffer().stats()?.is_constant().map(bool::into),
            Stat::IsSorted => self.flatbuffer().stats()?.is_sorted().map(bool::into),
            Stat::IsStrictSorted => self
                .flatbuffer()
                .stats()?
                .is_strict_sorted()
                .map(bool::into),
            Stat::RunCount => self.flatbuffer().stats()?.run_count().map(u64::into),
            Stat::TrueCount => self.flatbuffer().stats()?.true_count().map(u64::into),
            Stat::NullCount => self.flatbuffer().stats()?.null_count().map(u64::into),
            Stat::BitWidthFreq => {
                let element_dtype =
                    Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable));
                self.flatbuffer()
                    .stats()?
                    .bit_width_freq()
                    .map(|v| v.iter().map(Scalar::from).collect_vec())
                    .map(|v| Scalar::list(element_dtype, v))
            }
            Stat::TrailingZeroFreq => self
                .flatbuffer()
                .stats()?
                .trailing_zero_freq()
                .map(|v| v.iter().collect_vec())
                .map(|v| v.into()),
            Stat::UncompressedSizeInBytes => self
                .flatbuffer()
                .stats()?
                .uncompressed_size_in_bytes()
                .map(u64::into),
        }
    }

    /// NB: part of the contract for to_set is that it does not do any expensive computation.
    /// In other implementations, this means returning the underlying stats map, but for the flatbuffer
    /// implementation, we have 'precalculated' stats in the flatbuffer itself, so we need to
    /// allocate a stats map and populate it with those fields.
    fn to_set(&self) -> StatsSet {
        let mut result = StatsSet::default();
        for stat in all::<Stat>() {
            if let Some(value) = self.get(stat) {
                result.set(stat, value)
            }
        }
        result
    }

    /// We want to avoid any sort of allocation on instantiation of the ArrayView, so we
    /// do not allocate a stats_set to cache values.
    fn set(&self, _stat: Stat, _value: Scalar) {
        // We cannot modify stats on a view
    }

    fn clear(&self, _stat: Stat) {
        // We cannot modify stats on a view
    }

    fn retain_only(&self, _stats: &[Stat]) {
        // We cannot modify stats on a view
    }

    fn compute(&self, stat: Stat) -> Option<Scalar> {
        if let Some(s) = self.get(stat) {
            return Some(s);
        }

        self.encoding()
            .compute_statistics(&ArrayData::from(self.clone()), stat)
            .ok()?
            .get(stat)
            .cloned()
    }
}
