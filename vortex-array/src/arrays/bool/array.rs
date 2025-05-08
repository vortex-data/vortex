use arcref::ArcRef;
use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, MutableBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_panic};
use vortex_scalar::Scalar;

use crate::array::{Array, ArrayCanonicalImpl};
use crate::arrays::bool;
use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{VTable, ValidityChild, ValidityVTableFromValidityChild};
use crate::{ArrayRef, Canonical, Encoding, vtable};

vtable!(Bool);

#[derive(Clone, Debug)]
pub struct BoolArray {
    dtype: DType,
    buffer: BooleanBuffer,
    pub(crate) validity: Validity,
    pub(crate) stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct BoolEncoding;

impl VTable for BoolVTable {
    type Array = BoolArray;
    type Encoding = BoolEncoding;

    type ValidityVTable = ValidityVTableFromValidityChild;
    // Enable serde for this encoding
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> ArcRef<str> {
        ArcRef::new_ref("vortex.bool")
    }

    fn len(array: &Self::Array) -> usize {
        array.buffer.len()
    }

    fn dtype(array: &Self::Array) -> &DType {
        &array.dtype
    }

    fn stats(array: &Self::Array) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array)
    }

    fn slice(array: &Self::Array, start: usize, stop: usize) -> VortexResult<Self::Array> {
        todo!()
    }

    fn scalar_at(array: &Self::Array, index: usize) -> VortexResult<Scalar> {
        todo!()
    }

    fn with_children(array: &Self::Array, children: &[ArrayRef]) -> VortexResult<Self::Array> {
        let validity = if array.validity().is_array() {
            Validity::Array(children[0].clone())
        } else {
            array.validity().clone()
        };
        Ok(BoolArray::new(array.boolean_buffer().clone(), validity))
    }
}

impl BoolArray {
    /// Create a new BoolArray from a set of indices and a length.
    /// All indices must be less than the length.
    pub fn from_indices<I: IntoIterator<Item = usize>>(length: usize, indices: I) -> Self {
        let mut buffer = MutableBuffer::new_null(length);
        indices
            .into_iter()
            .for_each(|idx| arrow_buffer::bit_util::set_bit(&mut buffer, idx));
        Self::new(
            BooleanBufferBuilder::new_from_buffer(buffer, length).finish(),
            Validity::NonNullable,
        )
    }

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

impl From<BooleanBuffer> for BoolArray {
    fn from(value: BooleanBuffer) -> Self {
        Self::new(value, Validity::NonNullable)
    }
}

impl FromIterator<bool> for BoolArray {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::new(BooleanBuffer::from_iter(iter), Validity::NonNullable)
    }
}

impl FromIterator<Option<bool>> for BoolArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let (buffer, nulls) = BooleanArray::from_iter(iter).into_parts();

        Self::new(
            buffer,
            nulls.map(Validity::from).unwrap_or(Validity::AllValid),
        )
    }
}

impl ValidityChild for BoolArray {
    fn validity(&self) -> &Validity {
        &self.validity
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

pub trait BooleanBufferExt {
    /// Slice any full bytes from the buffer, leaving the offset < 8.
    fn shrink_offset(self) -> Self;
}

impl BooleanBufferExt for BooleanBuffer {
    fn shrink_offset(self) -> Self {
        let byte_offset = self.offset() / 8;
        let bit_offset = self.offset() % 8;
        let len = self.len();
        let buffer = self
            .into_inner()
            .slice_with_length(byte_offset, (len + bit_offset).div_ceil(8));
        BooleanBuffer::new(buffer, bit_offset, len)
    }
}
