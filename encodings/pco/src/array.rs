use std::cmp;
use std::fmt::Debug;

use pco::data_types::NumberType;
use pco::wrapped::{ChunkCompressor, FileCompressor};
use pco::{ChunkConfig, PagingSpec, match_number_enum};
use vortex_array::arrays::{PrimitiveArray, PrimitiveVTable};
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, vtable};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::PcoBufferMetadata;
use crate::serde::PcoMetadata;

// Overall approach here:
// Chunk the array into Pco chunks (currently using the default recommended size
// for good compression), and into finer-grained Pco pages. As we go, write each
// ChunkMeta as a buffer, followed by each of that chunk's pages as a buffer. We
// store metadata for each of these "components" (chunk or page). At
// decompression time, we figure out which components we need to read and only
// process those. We only compress and decompress valid values.

// Visually, during decompression, we have an interval of pages we're
// decompressing and a tighter interval of the slice we actually care about.
//
// |=============values (all valid elements)==============|
// |<-n_skipped_values->|----decompressed_values------|
//                          |----slice_values----|
//                          ^                    ^
// |<---slice_value_start-->|<--slice_n_values-->|
//
// We then insert these values to the correct position using a primitive array
// constructor.

const VALUES_PER_CHUNK: usize = pco::DEFAULT_MAX_PAGE_N;

vtable!(Pco);

impl VTable for PcoVTable {
    type Array = PcoArray;
    type Encoding = PcoEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.pco")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(PcoEncoding.as_ref())
    }
}

fn get_pco_type(ptype: PType) -> Option<NumberType> {
    let pco_type = match ptype {
        PType::F16 => NumberType::F16,
        PType::F32 => NumberType::F32,
        PType::F64 => NumberType::F64,
        PType::I16 => NumberType::I16,
        PType::I32 => NumberType::I32,
        PType::I64 => NumberType::I64,
        PType::U16 => NumberType::U16,
        PType::U32 => NumberType::U32,
        PType::U64 => NumberType::U64,
        _ => return None,
    };

    return Some(pco_type);
}

#[derive(Clone, Debug)]
pub struct PcoEncoding;

#[derive(Clone, Debug)]
enum PcoBufferKind {
    ChunkMeta,
    Page,
}

#[derive(Clone, Debug)]
pub struct PcoArray {
    pub(crate) buffers: Vec<ByteBuffer>,
    pub(crate) validity: Validity,
    pub(crate) metadata: PcoMetadata,
    dtype: DType,
    stats_set: ArrayStats,
    slice_start: usize,
    slice_stop: usize,
}

impl PcoArray {
    pub fn new(
        buffers: Vec<ByteBuffer>,
        dtype: DType,
        metadata: PcoMetadata,
        len: usize,
        validity: Validity,
    ) -> Self {
        Self {
            buffers,
            validity,
            metadata,
            dtype,
            stats_set: Default::default(),
            slice_start: 0,
            slice_stop: len,
        }
    }

    fn type_byte_width(&self) -> usize {
        self.dtype.as_ptype().byte_width()
    }

    pub fn uncompressed_size(&self) -> usize {
        (self.slice_stop - self.slice_start) * self.type_byte_width()
    }

    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: usize,
        nums_per_page: usize,
    ) -> VortexResult<Self> {
        Self::from_primitive_with_values_per_chunk(parray, level, VALUES_PER_CHUNK, values_per_page)
    }

    pub(crate) fn from_primitive_with_values_per_chunk(
        parray: &PrimitiveArray,
        level: usize,
        values_per_chunk: usize,
        values_per_page: usize,
    ) -> VortexResult<Self> {
        let number_type = number_type_from_dtype(parray.dtype())?;
        let values_per_page = if values_per_page == 0 {
            values_per_chunk
        } else {
            values_per_page
        };

        // perhaps one day we can make this more configurable
        let chunk_config = ChunkConfig::default()
            .with_compression_level(level)
            .with_paging_spec(PagingSpec::EqualPagesUpTo(nums_per_page));
        let mut start_n = 0;
        let mut buffers = vec![];
        let mut buffer_metas = vec![];

        let fc = FileCompressor::default();
        while start_n < n {
            let chunk_n = cmp::min(chunk_n, n - start_n);
            let stop_n = start_n + chunk_n;

        let mut chunk_meta_buffers = vec![]; // the Pco component
        let mut chunk_infos = vec![]; // the Vortex metadata
        let mut page_buffers = vec![];
        for chunk_start in (0..n_values).step_by(values_per_chunk) {
            let cc = match_number_enum!(
                number_type,
                NumberType<T> => {
                    let chunk_end = cmp::min(n_values, chunk_start + values_per_chunk);
                    let values = values.buffer::<T>();
                    let chunk = &values.as_slice()[chunk_start..chunk_end];
                    fc
                        .chunk_compressor(chunk, &chunk_config)
                        .map_err(vortex_err_from_pco)?
                }
            );
            let mut chunk_meta = vec![];
            cc.write_chunk_meta(&mut chunk_meta)?;
            buffers.push(ByteBuffer::from(chunk_meta));
            buffer_metas.push(PcoBufferMetadata {
                is_chunk_meta: true,
                n: chunk_n as u64,
            });

            for (i, page_n) in cc.n_per_page().into_iter().enumerate() {
                let mut page = vec![];
                cc.write_page(i, &mut page);
                buffers.push(ByteBuffer::from(page));
                buffer_metas.push(PcoBufferMetadata {
                    is_chunk_meta: false,
                    n: page_n as u64,
                });
            }

            start_n = stop_n;
        }

        let metadata = PcoMetadata {
            buffers: buffer_metas,
        };

        Ok(PcoArray::new(
            buffers,
            dtype,
            metadata,
            parray.len(),
            parray.validity().clone(),
        ))
    }

    pub fn from_array(array: ArrayRef, level: usize, nums_per_page: usize) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            Self::from_primitive(parray, level, nums_per_page)
        } else {
            Err(vortex_err!("Pco can only encode primitive arrays"))
        }
    }

    pub fn decompress(&self) -> VortexResult<ArrayRef> {
        // To start, we figure out which buffers we need to decompress, and with
        // what byte offset into the first such buffer.
        let type_size = self.type_byte_width()
        let mut buf_start = 0;
        let mut buffer_idx_lb = 0;
        let mut buffer_idx_ub = 0;
        let mut byte_offset = 0;
        for (i, buffer_meta) in self.metadata.buffers.iter().enumerate() {
            let buf_stop = buf_start + usize::try_from(buffer_meta.uncompressed_size)?;
            if buf_start < byte_start {
                buffer_idx_lb = i;
                byte_offset = byte_start - buf_start
            }
            if buf_start < byte_stop {
                buffer_idx_ub = i + 1
            }
            buf_start = buf_stop;
        }

        // then we actually decompress those buffers
        let buffer_metas = &self.metadata.buffers[buffer_idx_lb..buffer_idx_ub];
        let total_uncompressed_size: usize = buffer_metas
            .iter()
            .map(|meta| meta.uncompressed_size)
            .sum::<u64>()
            .try_into()?;

        let mut decompressor = if let Some(dictionary_buffer) = &self.chunk_meta_buffers {
            pco::bulk::Decompressor::with_dictionary(dictionary_buffer)
        } else {
            pco::bulk::Decompressor::new()
        }?;

        // we could make this empty initialized for better performance
        let mut decompressed = vec![0; total_uncompressed_size];
        let mut start = 0;
        for (buffer, meta) in self.page_buffers[buffer_idx_lb..buffer_idx_ub]
            .iter()
            .zip(buffer_metas)
        {
            let stop = start + usize::try_from(meta.uncompressed_size)?;
            decompressor.decompress_to_buffer(buffer.as_slice(), &mut decompressed[start..stop])?;
            start = stop;
        }

        // Last, we apply our byte offset. We need to copy since the
        // decompressed buffer start/end might not align with our slice.
        // And we need to align the data to our (dynamic) dtype.
        let slice_len = byte_stop - byte_start;
        let bytes = decompressed[byte_offset..byte_offset + slice_len].to_vec();
        let decompressed_buffer = ByteBuffer::from(bytes).aligned(Alignment::new(type_size));

        let primitive = PrimitiveArray::from_byte_buffer(
            decompressed_buffer,
            self.dtype.as_ptype(),
            self.validity.clone(),
        );

        Ok(primitive.into_array())
    }

    #[allow(clippy::unwrap_in_result, clippy::unwrap_used)]
    fn decompress_values_typed<T: Number>(&self) -> VortexResult<ByteBuffer> {
        // To start, we figure out which chunks and pages we need to decompress, and with
        // what value offset into the first such page.
        let slice_n_rows = self.slice_stop - self.slice_start;
        let mask = self.validity.to_mask(self.n_rows)?;
        let value_indices = mask.valid_counts_for_indices(&[self.slice_start, self.slice_stop])?;
        let slice_value_start = value_indices[0];
        let slice_value_stop = value_indices[1];

        let (fd, _) =
            FileDecompressor::new(self.metadata.header.as_slice()).map_err(vortex_err_from_pco)?;
        let mut decompressed_values = BufferMut::<T>::with_capacity(slice_n_rows);
        let mut page_idx = 0;
        let mut page_value_start = 0;
        let mut n_skipped_values = 0;
        for (chunk_info, chunk_meta) in self.metadata.chunks.iter().zip(&self.chunk_metas) {
            let mut cd: Option<ChunkDecompressor<T>> = None;
            for page_info in &chunk_info.pages {
                let page_n_values = page_info.n_values as usize;
                let page_value_stop = page_value_start + page_n_values;

                if page_value_start >= slice_value_stop {
                    break;
                }

                if page_value_stop > slice_value_start {
                    // we need this page
                    let old_len = decompressed_values.len();
                    let new_len = old_len + page_n_values;
                    decompressed_values.reserve(page_n_values);
                    unsafe {
                        decompressed_values.set_len(new_len);
                    }
                    let chunk_meta_bytes: &[u8] = chunk_meta.as_ref();
                    let page: &[u8] = self.pages[page_idx].as_ref();
                    if cd.is_none() {
                        let (new_cd, _) = fd
                            .chunk_decompressor(chunk_meta_bytes)
                            .map_err(vortex_err_from_pco)?;
                        cd = Some(new_cd);
                    }
                    let mut pd = cd
                        .as_mut()
                        .unwrap()
                        .page_decompressor(page, page_n_values)
                        .map_err(vortex_err_from_pco)?;
                    pd.decompress(&mut decompressed_values[old_len..new_len])
                        .map_err(vortex_err_from_pco)?;
                } else {
                    n_skipped_values += page_n_values;
                }

                page_value_start = page_value_stop;
                page_idx += 1;
            }
        }

        let value_offset = slice_value_start - n_skipped_values;
        Ok(decompressed_values
            .freeze()
            .slice(value_offset..value_offset + slice_n_rows)
            .into_byte_buffer())
    }

    fn _slice(&self, start: usize, stop: usize) -> Self {
        PcoArray {
            slice_start: self.slice_start + start,
            slice_stop: self.slice_start + stop,
            ..self.clone()
        }
    }
}

impl ValidityHelper for PcoArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<PcoVTable> for PcoVTable {
    fn len(array: &PcoArray) -> usize {
        array.slice_stop - array.slice_start
    }

    fn dtype(array: &PcoArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &PcoArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<PcoVTable> for PcoVTable {
    fn canonicalize(array: &PcoArray) -> VortexResult<Canonical> {
        array.decompress()?.to_canonical()
    }
}

impl OperationsVTable<PcoVTable> for PcoVTable {
    fn slice(array: &PcoArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        if array.slice_start + cmp::max(start, stop) > array.slice_stop {
            vortex_bail!("Cannot slice beyond end of PcoArray")
        }

        let sliced = PcoArray {
            slice_start: array.slice_start + start,
            slice_stop: array.slice_start + stop,
            validity: array.validity.slice(start, stop)?,
            ..array.clone()
        };
        Ok(sliced.into_array())
    }

    fn scalar_at(array: &PcoArray, index: usize) -> VortexResult<Scalar> {
        if array.slice_start + index >= array.slice_stop {
            vortex_bail!(
                "Cannot access {}th element from Pco slice [{}, {})",
                index,
                array.slice_start,
                array.slice_stop
            );
        }
        array.decompress()?.scalar_at(index)
    }
}
