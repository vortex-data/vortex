use std::fmt::Debug;

use vortex_array::arrays::{PrimitiveArray, PrimitiveVTable};
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, vtable};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::Scalar;

vtable!(Zstd);

impl VTable for ZstdVTable {
    type Array = ZstdArray;
    type Encoding = ZstdEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.zstd")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ZstdEncoding.as_ref())
    }
}

/// Zstd encoding with configurable parameters
#[derive(Clone, Debug)]
pub struct ZstdEncoding;

#[derive(Clone, Debug)]
pub struct ZstdArray {
    dtype: DType,
    compressed_data: ByteBuffer,
    pub(crate) uncompressed_len: usize,
    pub(crate) validity: Validity,
    stats_set: ArrayStats,
}

impl ZstdArray {
    pub fn compressed_data(&self) -> &ByteBuffer {
        &self.compressed_data
    }

    pub fn new(
        compressed_data: ByteBuffer,
        dtype: DType,
        uncompressed_len: usize,
        validity: Validity,
    ) -> Self {
        Self {
            dtype,
            compressed_data,
            uncompressed_len,
            validity,
            stats_set: Default::default(),
        }
    }

    pub fn from_primitive(parray: &PrimitiveArray, level: i32) -> VortexResult<Self> {
        let buffer = parray.byte_buffer();

        let compressed = zstd::bulk::compress(buffer.inner(), level)
            .map_err(|e| vortex_err!("Failed to compress array with zstd: {}", e))?;

        let compressed_buffer = ByteBuffer::from(compressed);
        let dtype = parray.dtype().clone();
        let uncompressed_len = parray.nbytes();

        Ok(ZstdArray::new(
            compressed_buffer,
            dtype,
            uncompressed_len,
            parray.validity().clone(),
        ))
    }

    pub fn try_from_array(array: ArrayRef, level: i32) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            Self::from_primitive(parray, level)
        } else {
            Err(vortex_err!("Zstd can only encode primitive arrays"))
        }
    }

    pub fn decompress(&self) -> VortexResult<ArrayRef> {
        let decompressed =
            zstd::bulk::decompress(self.compressed_data.as_slice(), self.uncompressed_len)
                .map_err(|e| vortex_err!("Failed to decompress zstd array: {}", e))?;

        let buffer = ByteBuffer::from(decompressed);

        let primitive =
            PrimitiveArray::from_byte_buffer(buffer, self.dtype.as_ptype(), self.validity.clone());

        Ok(primitive.into_array())
    }
}

impl ValidityHelper for ZstdArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<ZstdVTable> for ZstdVTable {
    fn len(array: &ZstdArray) -> usize {
        array.uncompressed_len
    }

    fn dtype(array: &ZstdArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ZstdArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<ZstdVTable> for ZstdVTable {
    fn canonicalize(array: &ZstdArray) -> VortexResult<Canonical> {
        array.decompress()?.to_canonical()
    }
}

impl OperationsVTable<ZstdVTable> for ZstdVTable {
    fn slice(array: &ZstdArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        array.decompress()?.slice(start, stop)
    }

    fn scalar_at(array: &ZstdArray, index: usize) -> VortexResult<Scalar> {
        array.decompress()?.scalar_at(index)
    }
}
