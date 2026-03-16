// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::fmt::Debug;
use std::hash::Hash;

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
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionStep;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::ValiditySliceHelper;
use vortex_array::vtable::ValidityVTableFromValiditySliceHelper;
use vortex_array::vtable::validity_nchildren;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

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

vtable!(Pco);

impl VTable for Pco {
    type Array = PcoArray;

    type Metadata = ProstMetadata<PcoMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValiditySliceHelper;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &PcoArray) -> usize {
        array.slice_stop - array.slice_start
    }

    fn dtype(array: &PcoArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &PcoArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &PcoArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.unsliced_validity.array_hash(state, precision);
        array.unsliced_n_rows.hash(state);
        array.slice_start.hash(state);
        array.slice_stop.hash(state);
        // Hash chunk_metas and pages using pointer-based hashing
        for chunk_meta in &array.chunk_metas {
            chunk_meta.array_hash(state, precision);
        }
        for page in &array.pages {
            page.array_hash(state, precision);
        }
    }

    fn array_eq(array: &PcoArray, other: &PcoArray, precision: Precision) -> bool {
        if array.dtype != other.dtype
            || !array
                .unsliced_validity
                .array_eq(&other.unsliced_validity, precision)
            || array.unsliced_n_rows != other.unsliced_n_rows
            || array.slice_start != other.slice_start
            || array.slice_stop != other.slice_stop
            || array.chunk_metas.len() != other.chunk_metas.len()
            || array.pages.len() != other.pages.len()
        {
            return false;
        }
        for (a, b) in array.chunk_metas.iter().zip(&other.chunk_metas) {
            if !a.array_eq(b, precision) {
                return false;
            }
        }
        for (a, b) in array.pages.iter().zip(&other.pages) {
            if !a.array_eq(b, precision) {
                return false;
            }
        }
        true
    }

    fn nbuffers(array: &PcoArray) -> usize {
        array.chunk_metas.len() + array.pages.len()
    }

    fn buffer(array: &PcoArray, idx: usize) -> BufferHandle {
        if idx < array.chunk_metas.len() {
            BufferHandle::new_host(array.chunk_metas[idx].clone())
        } else {
            let page_idx = idx - array.chunk_metas.len();
            BufferHandle::new_host(array.pages[page_idx].clone())
        }
    }

    fn buffer_name(array: &PcoArray, idx: usize) -> Option<String> {
        if idx < array.chunk_metas.len() {
            Some(format!("chunk_meta_{idx}"))
        } else {
            Some(format!("page_{}", idx - array.chunk_metas.len()))
        }
    }

    fn nchildren(array: &PcoArray) -> usize {
        validity_nchildren(&array.unsliced_validity)
    }

    fn child(array: &PcoArray, idx: usize) -> ArrayRef {
        validity_to_child(&array.unsliced_validity, array.unsliced_n_rows)
            .unwrap_or_else(|| vortex_panic!("PcoArray child index {idx} out of bounds"))
    }

    fn child_name(_array: &PcoArray, idx: usize) -> String {
        match idx {
            0 => "validity".to_string(),
            _ => vortex_panic!("PcoArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &PcoArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(array.metadata.clone()))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(PcoMetadata::decode(bytes)?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<PcoArray> {
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("PcoArray expected 0 or 1 child, got {}", children.len());
        };

        vortex_ensure!(buffers.len() >= metadata.0.chunks.len());
        let chunk_metas = buffers[..metadata.0.chunks.len()]
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;
        let pages = buffers[metadata.0.chunks.len()..]
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;

        let expected_n_pages = metadata
            .0
            .chunks
            .iter()
            .map(|info| info.pages.len())
            .sum::<usize>();
        vortex_ensure!(pages.len() == expected_n_pages);

        Ok(PcoArray::new(
            chunk_metas,
            pages,
            dtype.clone(),
            metadata.0.clone(),
            len,
            validity,
        ))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() <= 1,
            "PcoArray expects 0 or 1 children, got {}",
            children.len()
        );

        if children.is_empty() {
            array.unsliced_validity = Validity::from(array.dtype.nullability());
        } else {
            array.unsliced_validity =
                Validity::Array(children.into_iter().next().vortex_expect("validity child"));
        }

        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(array.decompress(ctx)?.into_array()))
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

pub(crate) fn number_type_from_dtype(dtype: &DType) -> NumberType {
    let ptype = dtype.as_ptype();
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

fn collect_valid(parray: &PrimitiveArray) -> VortexResult<PrimitiveArray> {
    let mask = parray.validity_mask()?;
    Ok(parray.clone().into_array().filter(mask)?.to_primitive())
}

pub(crate) fn vortex_err_from_pco(err: PcoError) -> VortexError {
    use pco::errors::ErrorKind::*;
    match err.kind {
        Io(io_kind) => VortexError::from(std::io::Error::new(io_kind, err.message)),
        InvalidArgument => vortex_err!(InvalidArgument: "{}", err.message),
        other => vortex_err!("Pco {:?} error: {}", other, err.message),
    }
}

#[derive(Debug)]
pub struct Pco;

impl Pco {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.pco");
}

#[derive(Clone, Debug)]
pub struct PcoArray {
    pub(crate) chunk_metas: Vec<ByteBuffer>,
    pub(crate) pages: Vec<ByteBuffer>,
    pub(crate) metadata: PcoMetadata,
    dtype: DType,
    pub(crate) unsliced_validity: Validity,
    unsliced_n_rows: usize,
    stats_set: ArrayStats,
    slice_start: usize,
    slice_stop: usize,
}

impl PcoArray {
    pub fn new(
        chunk_metas: Vec<ByteBuffer>,
        pages: Vec<ByteBuffer>,
        dtype: DType,
        metadata: PcoMetadata,
        len: usize,
        validity: Validity,
    ) -> Self {
        Self {
            chunk_metas,
            pages,
            metadata,
            dtype,
            unsliced_validity: validity,
            unsliced_n_rows: len,
            stats_set: Default::default(),
            slice_start: 0,
            slice_stop: len,
        }
    }

    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: usize,
        values_per_page: usize,
    ) -> VortexResult<Self> {
        Self::from_primitive_with_values_per_chunk(parray, level, VALUES_PER_CHUNK, values_per_page)
    }

    pub(crate) fn from_primitive_with_values_per_chunk(
        parray: &PrimitiveArray,
        level: usize,
        values_per_chunk: usize,
        values_per_page: usize,
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

        let values = collect_valid(parray)?;
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
        Ok(PcoArray::new(
            chunk_meta_buffers,
            page_buffers,
            parray.dtype().clone(),
            metadata,
            parray.len(),
            parray.validity().clone(),
        ))
    }

    pub fn from_array(array: ArrayRef, level: usize, nums_per_page: usize) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<Primitive>() {
            Self::from_primitive(parray, level, nums_per_page)
        } else {
            Err(vortex_err!("Pco can only encode primitive arrays"))
        }
    }

    pub fn decompress(&self, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        // To start, we figure out which chunks and pages we need to decompress, and with
        // what value offset into the first such page.
        let number_type = number_type_from_dtype(&self.dtype);
        let values_byte_buffer = match_number_enum!(
            number_type,
            NumberType<T> => {
              self.decompress_values_typed::<T>(ctx)?
            }
        );

        Ok(PrimitiveArray::from_values_byte_buffer(
            values_byte_buffer,
            self.dtype.as_ptype(),
            self.unsliced_validity
                .slice(self.slice_start..self.slice_stop)?,
            self.slice_stop - self.slice_start,
        ))
    }

    fn decompress_values_typed<T: Number>(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ByteBuffer> {
        // To start, we figure out what range of values we need to decompress.
        let slice_value_indices = self
            .unsliced_validity
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
        PcoArray {
            slice_start: self.slice_start + start,
            slice_stop: self.slice_start + stop,
            stats_set: Default::default(),
            ..self.clone()
        }
    }

    pub(crate) fn dtype(&self) -> &DType {
        &self.dtype
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

impl ValiditySliceHelper for PcoArray {
    fn unsliced_validity_and_slice(&self) -> (&Validity, usize, usize) {
        (&self.unsliced_validity, self.slice_start, self.slice_stop)
    }
}

impl OperationsVTable<Pco> for Pco {
    fn scalar_at(array: &PcoArray, index: usize) -> VortexResult<Scalar> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        array
            ._slice(index, index + 1)
            .decompress(&mut ctx)?
            .scalar_at(0)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::PcoArray;

    #[test]
    fn test_slice_nullable() {
        // Create a nullable array with some nulls
        let values = PrimitiveArray::new(
            buffer![10u32, 20, 30, 40, 50, 60],
            Validity::from_iter([false, true, true, true, true, false]),
        );
        let pco = PcoArray::from_primitive(&values, 0, 128).unwrap();
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
