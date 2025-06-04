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
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
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
    dtype: DType,
    stats_set: ArrayStats,
    slice_start: usize,
    slice_stop: usize,
}

fn choose_max_dict_size(uncompressed_size: usize) -> usize {
    // following recommendations from
    // https://github.com/facebook/zstd/blob/v1.5.5/lib/zdict.h#L190
    // that is, 1/100 the data size, up to 100kB.
    // It appears that zstd can't train dictionaries with <256 bytes.
    (uncompressed_size / 100).clamp(256, 100 * 1024)
}

impl ZstdArray {
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
            dtype,
            stats_set: Default::default(),
            slice_start: 0,
            slice_stop: len,
        }
    }

    pub fn uncompressed_size(&self) -> usize {
        (self.slice_stop - self.slice_start) * self.dtype.as_ptype().byte_width()
    }

    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: i32,
        rows_per_buffer: usize,
    ) -> VortexResult<Self> {
        let buffer = parray.byte_buffer();
        let bytes = buffer.inner();
        let type_size = parray.ptype().byte_width();

        let (dictionary_buffer, mut compressor, chunk_byte_size) =
            if rows_per_buffer == 0 || rows_per_buffer >= parray.len() {
                // single buffer, no dictionary
                (
                    None,
                    zstd::bulk::Compressor::new(level)?,
                    parray.len() * type_size,
                )
            } else {
                // multi buffer, with dictionary
                let chunk_byte_size = rows_per_buffer * type_size;
                let sample_sizes = bytes
                    .chunks(chunk_byte_size)
                    .map(|chunk| chunk.len())
                    .collect::<Vec<_>>();
                debug_assert!(sample_sizes.iter().sum::<usize>() == bytes.len());
                let dict = zstd::dict::from_continuous(
                    bytes,
                    &sample_sizes,
                    choose_max_dict_size(buffer.len()),
                )
                .map_err(|err| {
                    Into::<VortexError>::into(err).with_context("while training dictionary")
                })?;

                let compressor = zstd::bulk::Compressor::with_dictionary(level, &dict)?;
                (Some(ByteBuffer::from(dict)), compressor, chunk_byte_size)
            };

        let mut compressed_buffer_metas = vec![];
        let mut compressed_buffers = vec![];
        for chunk in buffer.chunks(chunk_byte_size) {
            let compressed = compressor
                .compress(chunk)
                .map_err(|err| Into::<VortexError>::into(err).with_context("while compressing"))?;
            compressed_buffer_metas.push(ZstdBufferMetadata {
                compressed_size: compressed.len() as u64,
                uncompressed_size: chunk.len() as u64,
            });
            compressed_buffers.push(ByteBuffer::from(compressed));
        }

        let dtype = parray.dtype().clone();
        let metadata = ZstdMetadata {
            dictionary_size: dictionary_buffer
                .as_ref()
                .map_or(0, |buffer| buffer.len())
                .try_into()?,
            compressed_buffers: compressed_buffer_metas,
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
        let type_size = self.dtype.as_ptype().byte_width();
        let byte_start = self.slice_start * type_size;
        let byte_stop = self.slice_stop * type_size;
        let mut buf_start = 0;
        let mut buffer_idx_lb = 0;
        let mut buffer_idx_ub = 0;
        let mut byte_offset = 0;
        for (i, buffer_meta) in self.metadata.compressed_buffers.iter().enumerate() {
            let buf_stop = buf_start + buffer_meta.uncompressed_size as usize;
            if buf_start < byte_start {
                buffer_idx_lb = i;
                byte_offset = byte_start - buf_start
            }
            if buf_start < byte_stop {
                buffer_idx_ub = i + 1
            }
            buf_start = buf_stop;
        }

        let buffer_metas = &self.metadata.compressed_buffers[buffer_idx_lb..buffer_idx_ub];
        let total_uncompressed_size = buffer_metas
            .iter()
            .map(|meta| meta.uncompressed_size)
            .sum::<u64>() as usize;

        let mut decompressor = if let Some(dictionary_buffer) = &self.dictionary_buffer {
            zstd::bulk::Decompressor::with_dictionary(&dictionary_buffer)
        } else {
            zstd::bulk::Decompressor::new()
        }?;
        // we could make this empty initialized for better performance
        let mut decompressed = vec![0; total_uncompressed_size];
        let mut start = 0;
        for (buffer, meta) in self.compressed_buffers[buffer_idx_lb..buffer_idx_ub]
            .iter()
            .zip(buffer_metas)
        {
            let stop = start + meta.uncompressed_size as usize;
            decompressor.decompress_to_buffer(buffer.as_slice(), &mut decompressed[start..stop])?;
            start = stop;
        }
        // Here we need to copy since the decompressed buffer start/end might not align
        // with our slice.
        let decompressed_buffer = ByteBuffer::from(
            decompressed[byte_offset..byte_offset + byte_stop - byte_start].to_vec(),
        );

        let primitive = PrimitiveArray::from_byte_buffer(
            decompressed_buffer,
            self.dtype.as_ptype(),
            self.validity.clone(),
        );

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
        array.slice_stop - array.slice_start
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
        if array.slice_start + index >= array.slice_stop {
            vortex_bail!(
                "Cannot access {}th element from Zstd slice [{}, {})",
                index,
                array.slice_start,
                array.slice_stop
            );
        }
        array.decompress()?.scalar_at(index)
    }
}
