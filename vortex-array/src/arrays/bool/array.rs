use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, MutableBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;

use crate::array::{Array, ArrayCanonicalImpl, ArrayValidityImpl, ArrayVariantsImpl};
use crate::arrays::bool;
use crate::arrays::bool::serde::BoolMetadata;
use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::variants::BoolArrayTrait;
use crate::vtable::{EncodingVTable, VTableRef};
use crate::{ArrayImpl, ArrayStatisticsImpl, Canonical, Encoding, EncodingId, RkyvMetadata};

#[derive(Clone, Debug)]
pub struct BoolArray {
    dtype: DType,
    buffer: BooleanBuffer,
    pub(crate) validity: Validity,
    // TODO(ngates): do we want a stats set to be shared across all arrays?
    pub(crate) stats_set: ArrayStats,
}

pub struct BoolEncoding;
impl Encoding for BoolEncoding {
    type Array = BoolArray;
    type Metadata = RkyvMetadata<BoolMetadata>;
}

impl EncodingVTable for BoolEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.bool")
    }
}

impl BoolArray {
    /// Creates a new [`BoolArray`] from a [`BooleanBuffer`] and [`Validity`], without checking
    /// any invariants.
    pub fn new(buffer: BooleanBuffer, validity: Validity) -> Self {
        if let Some(len) = validity.maybe_len() {
            if buffer.len() != len {
                vortex_panic!(
                    "Buffer and validity length mismatch: buffer={}, validity={}",
                    buffer.len(),
                    len
                );
            }
        }

        // Shrink the buffer to remove any whole bytes.
        let buffer = buffer.shrink_offset();

        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats_set: ArrayStats::default(),
        }
    }

    /// Returns the underlying [`BooleanBuffer`] of the array.
    pub fn boolean_buffer(&self) -> &BooleanBuffer {
        assert!(
            self.buffer.offset() < 8,
            "Offset must be <8, did we forget to call shrink_offset? Found {}",
            self.buffer.offset()
        );
        &self.buffer
    }

    /// Returns the underlying [`Validity`] of the array.
    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    /// Get a mutable version of this array.
    ///
    /// If the caller holds the only reference to the underlying buffer the underlying buffer is returned
    /// otherwise a copy is created.
    ///
    /// The second value of the tuple is a bit_offset of first value in first byte of the returned builder
    pub fn into_boolean_builder(self) -> (BooleanBufferBuilder, usize) {
        let offset = self.buffer.offset();
        let len = self.buffer.len();
        let arrow_buffer = self.buffer.into_inner();
        let mutable_buf = if arrow_buffer.ptr_offset() == 0 {
            arrow_buffer.into_mutable().unwrap_or_else(|b| {
                let mut buf = MutableBuffer::with_capacity(b.len());
                buf.extend_from_slice(b.as_slice());
                buf
            })
        } else {
            let mut buf = MutableBuffer::with_capacity(arrow_buffer.len());
            buf.extend_from_slice(arrow_buffer.as_slice());
            buf
        };

        (
            BooleanBufferBuilder::new_from_buffer(mutable_buf, offset + len),
            offset,
        )
    }
}

impl ArrayImpl for BoolArray {
    type Encoding = BoolEncoding;

    #[inline]
    fn _len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&BoolEncoding)
    }
}

impl ArrayCanonicalImpl for BoolArray {
    #[inline]
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::Bool(self.clone()))
    }

    #[inline]
    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        builder.extend_from_array(self)
    }
}

impl ArrayStatisticsImpl for BoolArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for BoolArray {
    #[inline]
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    #[inline]
    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    #[inline]
    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    #[inline]
    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_logical(self.len())
    }
}

impl ArrayVariantsImpl for BoolArray {
    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for BoolArray {}

pub trait BooleanBufferExt {
    /// Slice any full bytes from the buffer, leaving the offset < 8.
    fn shrink_offset(self) -> Self;
}

impl BooleanBufferExt for BooleanBuffer {
    fn shrink_offset(self) -> Self {
        let byte_offset = self.offset() / 8;
        let bit_offset = self.offset() % 8;
        let len = self.len();
        let buffer = self.into_inner().slice(byte_offset);
        BooleanBuffer::new(buffer, bit_offset, len)
    }
}
