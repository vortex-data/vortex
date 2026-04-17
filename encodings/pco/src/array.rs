// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use pco::ChunkConfig;
use pco::PagingSpec;
use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::errors::PcoError;
use pco::match_number_enum;
use pco::wrapped::ChunkDecompressor;
use pco::wrapped::FileCompressor;
use pco::wrapped::FileDecompressor;
use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::PcoChunkInfo;
use crate::PcoMetadata;
use crate::PcoPageInfo;

// Overall approach here:
// Chunk the array into Pco chunks (currently using the default recommended size
// for good compression), and into finer-grained Pco pages. As we go, write each
// ChunkMeta as a buffer, followed by each of that chunk's pages as a buffer. We
// store metadata for each of these "components" (chunk or page). At
// decompression time, we figure out which components we need to read and only
// process those. We only compress and decompress valid values.

// Visually, during decompression, we have an interval of pages we're
// decompressing and a tighter interval of the slice we actually care about.
// |=============values (all valid elements)==============|
// |<-n_skipped_values->|----decompressed_values------|
//                          |----slice_values----|
//                          ^                    ^
// |<---slice_value_start-->|<--slice_n_values-->|
// We then insert these values to the correct position using a primitive array
// constructor.

const VALUES_PER_CHUNK: usize = pco::DEFAULT_MAX_PAGE_N;

/// A [`Pco`]-encoded Vortex array.
pub type PcoArray = Array<Pco>;

impl ArrayHash for PcoData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.unsliced_n_rows.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
        // Hash chunk_metas and pages using pointer-based hashing
        for chunk_meta in &self.chunk_metas {
            chunk_meta.array_hash(state, precision);
        }
        for page in &self.pages {
            page.array_hash(state, precision);
        }
    }
}

impl ArrayEq for PcoData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        if self.unsliced_n_rows != other.unsliced_n_rows
            || self.slice_start != other.slice_start
            || self.slice_stop != other.slice_stop
            || self.chunk_metas.len() != other.chunk_metas.len()
            || self.pages.len() != other.pages.len()
        {
            return false;
        }
        for (a, b) in self.chunk_metas.iter().zip(&other.chunk_metas) {
            if !a.array_eq(b, precision) {
                return false;
            }
        }
        for (a, b) in self.pages.iter().zip(&other.pages) {
            if !a.array_eq(b, precision) {
                return false;
            }
        }
        true
    }
}

impl VTable for Pco {
    type ArrayData = PcoData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.pco");
        *ID
    }

    fn validate(
        &self,
        data: &PcoData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let validity = child_to_validity(&slots[0], dtype.nullability());
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.chunk_metas.len() + array.pages.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        if idx < array.chunk_metas.len() {
            BufferHandle::new_host(array.chunk_metas[idx].clone())
        } else {
            let page_idx = idx - array.chunk_metas.len();
            BufferHandle::new_host(array.pages[page_idx].clone())
        }
    }

    fn buffer_name(array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        if idx < array.chunk_metas.len() {
            Some(format!("chunk_meta_{idx}"))
        } else {
            Some(format!("page_{}", idx - array.chunk_metas.len()))
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.clone().encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = PcoMetadata::decode(metadata)?;
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("PcoArray expected 0 or 1 child, got {}", children.len());
        };

        vortex_ensure!(buffers.len() >= metadata.chunks.len());
        let chunk_metas = buffers[..metadata.chunks.len()]
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;
        let pages = buffers[metadata.chunks.len()..]
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;

        let expected_n_pages = metadata
            .chunks
            .iter()
            .map(|info| info.pages.len())
            .sum::<usize>();
        vortex_ensure!(pages.len() == expected_n_pages);

        let slots = vec![validity_to_child(&validity, len)];
        let data = PcoData::new(chunk_metas, pages, dtype.as_ptype(), metadata, len);
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let unsliced_validity =
            child_to_validity(&array.as_ref().slots()[0], array.dtype().nullability());
        Ok(ExecutionResult::done(
            array
                .data()
                .decompress(&unsliced_validity, ctx)?
                .into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

pub(crate) fn number_type_from_dtype(dtype: &DType) -> NumberType {
    number_type_from_ptype(dtype.as_ptype())
}

pub(crate) fn number_type_from_ptype(ptype: PType) -> NumberType {
    match ptype {
        PType::F16 => NumberType::F16,
        PType::F32 => NumberType::F32,
        PType::F64 => NumberType::F64,
        PType::I16 => NumberType::I16,
        PType::I32 => NumberType::I32,
        PType::I64 => NumberType::I64,
        PType::U16 => NumberType::U16,
        PType::U32 => NumberType::U32,
        PType::U64 => NumberType::U64,
        _ => unreachable!("PType not supported by Pco: {:?}", ptype),
    }
}

fn collect_valid(
    parray: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mask = parray
        .array()
        .validity()?
        .to_mask(parray.array().len(), ctx)?;
    let result = parray
        .array()
        .filter(mask)?
        .execute::<PrimitiveArray>(ctx)?;
    Ok(result)
}

pub(crate) fn vortex_err_from_pco(err: PcoError) -> VortexError {
    use pco::errors::ErrorKind::*;
    match err.kind {
        Io(io_kind) => VortexError::from(std::io::Error::new(io_kind, err.message)),
        InvalidArgument => vortex_err!(InvalidArgument: "{}", err.message),
        other => vortex_err!("Pco {:?} error: {}", other, err.message),
    }
}

#[derive(Clone, Debug)]
pub struct Pco;

impl Pco {
    pub(crate) fn try_new(
        dtype: DType,
        data: PcoData,
        validity: Validity,
    ) -> VortexResult<PcoArray> {
        let len = data.len();
        data.validate(&dtype, len, &validity)?;
        let slots = vec![validity_to_child(&validity, data.unsliced_n_rows())];
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Pco, dtype, len, data).with_slots(slots))
        })
    }

    /// Compress a primitive array using pcodec.
    pub fn from_primitive(
        parray: ArrayView<'_, Primitive>,
        level: usize,
        values_per_page: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PcoArray> {
        let dtype = parray.dtype().clone();
        let validity = parray.validity()?;
        let data = PcoData::from_primitive(parray, level, values_per_page, ctx)?;
        Self::try_new(dtype, data, validity)
    }
}

/// The validity bitmap indicating which elements are non-null.
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity"];

#[derive(Clone, Debug)]
pub struct PcoData {
    pub(crate) chunk_metas: Vec<ByteBuffer>,
    pub(crate) pages: Vec<ByteBuffer>,
    pub(crate) metadata: PcoMetadata,
    ptype: PType,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
}

impl Display for PcoData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ptype: {}, nrows: {}, slice: {}..{}",
            self.ptype, self.unsliced_n_rows, self.slice_start, self.slice_stop
        )
    }
}

impl PcoData {
    pub fn validate(&self, dtype: &DType, len: usize, validity: &Validity) -> VortexResult<()> {
        let _ = number_type_from_ptype(self.ptype);
        vortex_ensure!(
            dtype.as_ptype() == self.ptype,
            "expected ptype {}, got {}",
            self.ptype,
            dtype.as_ptype()
        );
        vortex_ensure!(
            dtype.nullability() == validity.nullability(),
            "expected nullability {}, got {}",
            validity.nullability(),
            dtype.nullability()
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop && self.slice_stop <= self.unsliced_n_rows,
            "invalid slice range {}..{} for {} rows",
            self.slice_start,
            self.slice_stop,
            self.unsliced_n_rows
        );
        vortex_ensure!(
            self.slice_stop - self.slice_start == len,
            "expected len {len}, got {}",
            self.slice_stop - self.slice_start
        );
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_n_rows,
                "expected validity len {}, got {}",
                self.unsliced_n_rows,
                validity_len
            );
        }
        vortex_ensure!(
            self.chunk_metas.len() == self.metadata.chunks.len(),
            "expected {} chunk metas, got {}",
            self.metadata.chunks.len(),
            self.chunk_metas.len()
        );
        vortex_ensure!(
            self.pages.len()
                == self
                    .metadata
                    .chunks
                    .iter()
                    .map(|chunk| chunk.pages.len())
                    .sum::<usize>(),
            "page count does not match metadata"
        );
        Ok(())
    }

    pub fn new(
        chunk_metas: Vec<ByteBuffer>,
        pages: Vec<ByteBuffer>,
        ptype: PType,
        metadata: PcoMetadata,
        len: usize,
    ) -> Self {
        Self {
            chunk_metas,
            pages,
            metadata,
            ptype,
            unsliced_n_rows: len,
            slice_start: 0,
            slice_stop: len,
        }
    }

    pub fn from_primitive(
        parray: ArrayView<'_, Primitive>,
        level: usize,
        values_per_page: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        Self::from_primitive_with_values_per_chunk(
            parray,
            level,
            VALUES_PER_CHUNK,
            values_per_page,
            ctx,
        )
    }

    pub(crate) fn from_primitive_with_values_per_chunk(
        parray: ArrayView<'_, Primitive>,
        level: usize,
        values_per_chunk: usize,
        values_per_page: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let number_type = number_type_from_dtype(parray.dtype());
        let values_per_page = if values_per_page == 0 {
            values_per_chunk
        } else {
            values_per_page
        };

        // perhaps one day we can make this more configurable
        let chunk_config = ChunkConfig::default()
            .with_compression_level(level)
            .with_paging_spec(PagingSpec::EqualPagesUpTo(values_per_page));

        let values = collect_valid(parray, ctx)?;
        let n_values = values.len();

        let fc = FileCompressor::default();
        let mut header = vec![];
        fc.write_header(&mut header).map_err(vortex_err_from_pco)?;

        let mut chunk_meta_buffers = vec![]; // the Pco component
        let mut chunk_infos = vec![]; // the Vortex metadata
        let mut page_buffers = vec![];
        for chunk_start in (0..n_values).step_by(values_per_chunk) {
            let chunk_end = cmp::min(n_values, chunk_start + values_per_chunk);
            let mut cc = match_number_enum!(
                number_type,
                NumberType<T> => {
                    let values = values.to_buffer::<T>();
                    let chunk = &values.as_slice()[chunk_start..chunk_end];
                    fc
                        .chunk_compressor(chunk, &chunk_config)
                        .map_err(vortex_err_from_pco)?
                }
            );

            let mut chunk_meta_buffer = ByteBufferMut::with_capacity(cc.meta_size_hint());
            cc.write_meta(&mut chunk_meta_buffer)
                .map_err(vortex_err_from_pco)?;
            chunk_meta_buffers.push(chunk_meta_buffer.freeze());

            let mut page_infos = vec![];
            for (page_idx, page_n_values) in cc.n_per_page().into_iter().enumerate() {
                let mut page = ByteBufferMut::with_capacity(cc.page_size_hint(page_idx));
                cc.write_page(page_idx, &mut page)
                    .map_err(vortex_err_from_pco)?;
                page_buffers.push(page.freeze());
                page_infos.push(PcoPageInfo {
                    n_values: u32::try_from(page_n_values)?,
                });
            }
            chunk_infos.push(PcoChunkInfo { pages: page_infos })
        }

        let metadata = PcoMetadata {
            header,
            chunks: chunk_infos,
        };
        Ok(PcoData::new(
            chunk_meta_buffers,
            page_buffers,
            parray.dtype().as_ptype(),
            metadata,
            parray.len(),
        ))
    }

    pub fn from_array(
        array: ArrayRef,
        level: usize,
        nums_per_page: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let parray = array.try_downcast::<Primitive>().map_err(|a| {
            vortex_err!(
                "Pco can only encode primitive arrays, got {}",
                a.encoding_id()
            )
        })?;
        Self::from_primitive(parray.as_view(), level, nums_per_page, ctx)
    }

    pub fn decompress(
        &self,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        // To start, we figure out which chunks and pages we need to decompress, and with
        // what value offset into the first such page.
        let number_type = number_type_from_ptype(self.ptype);
        let values_byte_buffer = match_number_enum!(
            number_type,
            NumberType<T> => {
              self.decompress_values_typed::<T>(unsliced_validity, ctx)?
            }
        );

        Ok(PrimitiveArray::from_values_byte_buffer(
            values_byte_buffer,
            self.ptype,
            unsliced_validity.slice(self.slice_start..self.slice_stop)?,
            self.slice_stop - self.slice_start,
        ))
    }

    fn decompress_values_typed<T: Number>(
        &self,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ByteBuffer> {
        // To start, we figure out what range of values we need to decompress.
        let slice_value_indices = unsliced_validity
            .execute_mask(self.unsliced_n_rows, ctx)?
            .valid_counts_for_indices(&[self.slice_start, self.slice_stop]);
        let slice_value_start = slice_value_indices[0];
        let slice_value_stop = slice_value_indices[1];
        let slice_n_values = slice_value_stop - slice_value_start;

        // Then we decompress those pages into a buffer. Note that these values
        // may exceed the bounds of the slice, so we need to slice later.
        let (fd, _) =
            FileDecompressor::new(self.metadata.header.as_slice()).map_err(vortex_err_from_pco)?;
        let mut decompressed_values = BufferMut::<T>::with_capacity(slice_n_values);
        let mut page_idx = 0;
        let mut page_value_start = 0;
        let mut n_skipped_values = 0;
        for (chunk_info, chunk_meta) in self.metadata.chunks.iter().zip(&self.chunk_metas) {
            // lazily initialize chunk decompressor
            let mut chunk_decompressor: Option<ChunkDecompressor<T>> = None;
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
                    let page: &[u8] = self.pages[page_idx].as_ref();

                    let mut cd = match chunk_decompressor.take() {
                        Some(d) => d,
                        None => {
                            let (new_cd, _) = fd
                                .chunk_decompressor(chunk_meta.as_ref())
                                .map_err(vortex_err_from_pco)?;
                            new_cd
                        }
                    };

                    let mut pd = cd
                        .page_decompressor(page, page_n_values)
                        .map_err(vortex_err_from_pco)?;
                    pd.read(&mut decompressed_values[old_len..new_len])
                        .map_err(vortex_err_from_pco)?;

                    chunk_decompressor = Some(cd);
                } else {
                    n_skipped_values += page_n_values;
                }

                page_value_start = page_value_stop;
                page_idx += 1;
            }
        }

        // Slice only the values requested.
        let value_offset = slice_value_start - n_skipped_values;
        Ok(decompressed_values
            .freeze()
            .slice(value_offset..value_offset + slice_n_values)
            .into_byte_buffer())
    }

    pub(crate) fn _slice(&self, start: usize, stop: usize) -> Self {
        PcoData {
            slice_start: self.slice_start + start,
            slice_stop: self.slice_start + stop,
            ..self.clone()
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.slice_stop == self.slice_start
    }

    pub(crate) fn slice_start(&self) -> usize {
        self.slice_start
    }

    pub(crate) fn slice_stop(&self) -> usize {
        self.slice_stop
    }

    pub(crate) fn unsliced_n_rows(&self) -> usize {
        self.unsliced_n_rows
    }
}

impl ValidityVTable<Pco> for Pco {
    fn validity(array: ArrayView<'_, Pco>) -> VortexResult<Validity> {
        let unsliced_validity = child_to_validity(&array.slots()[0], array.dtype().nullability());
        unsliced_validity.slice(array.slice_start()..array.slice_stop())
    }
}

impl OperationsVTable<Pco> for Pco {
    fn scalar_at(
        array: ArrayView<'_, Pco>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let unsliced_validity = child_to_validity(&array.slots()[0], array.dtype().nullability());
        array
            ._slice(index, index + 1)
            .decompress(&unsliced_validity, ctx)?
            .into_array()
            .execute_scalar(0, ctx)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::Pco;

    #[test]
    fn test_slice_nullable() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Create a nullable array with some nulls
        let values = PrimitiveArray::new(
            buffer![10u32, 20, 30, 40, 50, 60],
            Validity::from_iter([false, true, true, true, true, false]),
        );
        let pco = Pco::from_primitive(values.as_view(), 0, 128, &mut ctx).unwrap();
        assert_arrays_eq!(
            pco,
            PrimitiveArray::from_option_iter([
                None,
                Some(20u32),
                Some(30),
                Some(40),
                Some(50),
                None
            ])
        );

        // Slice to get only the non-null values in the middle
        let sliced = pco.slice(1..5).unwrap();
        let expected =
            PrimitiveArray::from_option_iter([Some(20u32), Some(30), Some(40), Some(50)])
                .into_array();
        assert_arrays_eq!(sliced, expected);
    }
}
