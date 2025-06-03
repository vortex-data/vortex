use std::cmp;
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
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::serde::{ZstdBufferMetadata, ZstdMetadata};

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

#[derive(Clone, Debug)]
pub struct ZstdEncoding;

#[derive(Clone, Debug)]
pub struct ZstdArray {
    pub(crate) dictionary_buffer: Option<ByteBuffer>,
    pub(crate) compressed_buffers: Vec<ByteBuffer>,
    pub(crate) validity: Validity,
    pub(crate) metadata: ZstdMetadata,
    pub(crate) len: usize,
    dtype: DType,
    stats_set: ArrayStats,
    slice_start: usize,
    slice_stop: usize,
}

fn choose_max_dict_size(uncompressed_size: usize) -> usize {
    // one day we could change this heuristic or make it configurable
    cmp::min(uncompressed_size / 100, 8192)
}

impl ZstdArray {
    fn compressed_buffers(&self) -> &[ByteBuffer] {
        &self.compressed_buffers
    }

    pub fn new(
        dictionary_buffer: Option<ByteBuffer>,
        compressed_buffers: Vec<ByteBuffer>,
        dtype: DType,
        metadata: ZstdMetadata,
        len: usize,
        validity: Validity,
    ) -> Self {
        Self {
            dictionary_buffer,
            compressed_buffers,
            validity,
            metadata,
            len,
            dtype,
            stats_set: Default::default(),
            slice_start: 0,
            slice_stop: len,
        }
    }

    pub fn uncompressed_size(&self) -> usize {
        self.len * self.dtype.as_ptype().byte_width()
    }

    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: i32,
        rows_per_buffer: usize,
    ) -> VortexResult<Self> {
        let buffer = parray.byte_buffer();
        let bytes = buffer.inner();
        let type_size = parray.ptype().byte_width();

        let (dictionary_buffer, compressed_buffer_metadatas, compressed_buffers) =
            if rows_per_buffer == 0 || rows_per_buffer >= parray.len() {
                // single buffer, no dictionary
                let compressed = zstd::bulk::compress(bytes, level)
                    .map_err(|e| vortex_err!("Failed to compress array with zstd: {}", e))?;
                (
                    None,
                    vec![ZstdBufferMetadata {
                        compressed_size: compressed.len() as u64,
                        uncompressed_size: buffer.len() as u64,
                    }],
                    vec![ByteBuffer::from(compressed)],
                )
            } else {
                // multi buffer, with dictionary
                let mut sample_sizes = vec![];
                let mut start = 0;
                while start < buffer.len() {
                    let stop = cmp::min(start + rows_per_buffer * type_size, buffer.len());
                    sample_sizes.push(stop - start);
                    start = stop;
                }
                let dict = zstd::dict::from_continuous(
                    bytes,
                    &sample_sizes,
                    choose_max_dict_size(buffer.len()),
                )?;

                let mut buffer_metas = vec![];
                let mut compressed_buffers = vec![];
                let mut start = 0;
                let mut compressor = zstd::bulk::Compressor::with_dictionary(level, &dict)?;
                while start < buffer.len() {
                    let stop = cmp::min(start + rows_per_buffer * type_size, buffer.len());
                    let compressed = compressor.compress(&bytes[start..stop])?;
                    buffer_metas.push(ZstdBufferMetadata {
                        compressed_size: compressed.len() as u64,
                        uncompressed_size: (stop - start) as u64,
                    });
                    compressed_buffers.push(ByteBuffer::from(compressed));
                    start = stop;
                }

                (
                    Some(ByteBuffer::from(dict)),
                    buffer_metas,
                    compressed_buffers,
                )
            };

        let dtype = parray.dtype().clone();
        let metadata = ZstdMetadata {
            dictionary_size: dictionary_buffer
                .as_ref()
                .map_or(0, |buffer| buffer.len())
                .try_into()?,
            buffers: compressed_buffer_metadatas,
            rows_per_buffer: rows_per_buffer.try_into()?,
        };

        Ok(ZstdArray::new(
            dictionary_buffer,
            compressed_buffers,
            dtype,
            metadata,
            parray.len(),
            parray.validity().clone(),
        ))
    }

    pub fn from_array(array: ArrayRef, level: i32, rows_per_buffer: usize) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            Self::from_primitive(parray, level, rows_per_buffer)
        } else {
            Err(vortex_err!("Zstd can only encode primitive arrays"))
        }
    }

    pub fn decompress(&self) -> VortexResult<ArrayRef> {
        let total_uncompressed_size = self
            .metadata
            .buffers
            .iter()
            .map(|meta| meta.uncompressed_size)
            .sum::<u64>() as usize;
        // we could make this empty initialized for better performance later
        let mut decompressed = vec![0; total_uncompressed_size];
        let mut start = 0;
        for (buffer, meta) in self.compressed_buffers.iter().zip(&self.metadata.buffers) {
            let stop = start + meta.uncompressed_size as usize;
            zstd::bulk::decompress_to_buffer(buffer.as_slice(), &mut decompressed[start..stop])
                .map_err(|e| vortex_err!("Failed to decompress zstd array: {}", e))?;
            start = stop;
        }

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
        array.len
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
        if array.slice_start + cmp::max(start, stop) > array.slice_stop {
            vortex_bail!("Cannot slice beyond end of ZstdArray")
        }

        let sliced = ZstdArray {
            slice_start: array.slice_start + start,
            slice_stop: array.slice_start + stop,
            ..array.clone()
        };
        Ok(sliced.into_array())
    }

    fn scalar_at(array: &ZstdArray, index: usize) -> VortexResult<Scalar> {
        let idx = array.slice_start + index;
        if idx >= array.slice_stop {
            vortex_bail!(
                "Cannot access {}th element from Zstd slice [{}, {})",
                index,
                array.slice_start,
                array.slice_stop
            );
        }
        array.decompress()?.scalar_at(array.slice_start + index)
    }
}
