// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use futures::pin_mut;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

#[derive(Clone)]
pub struct RepartitionWriterOptions {
    /// The minimum uncompressed size in bytes for a block.
    pub block_size_minimum: u64,
    /// The multiple of the number of rows in each block.
    pub block_len_multiple: usize,
    /// Optional target uncompressed size in bytes for a block.
    ///
    /// The repartition writer attempts to produce partitions with this uncompressed size. This is
    /// only a best effort attempt: the partitions may be arbitrarily larger or smaller. Reasons for
    /// this include:
    ///
    /// 1. The size of one element may not perfectly divide the target size, resulting in blocks
    ///    that are either too large or too small.
    ///
    /// 2. Variable length types are expensive to pack due to the need to read each element length.
    ///
    /// 3. View types are expensive to pack due to each view sharing an arbitrary slice of data.
    pub block_size_target: Option<u64>,
    pub canonicalize: bool,
}

impl RepartitionWriterOptions {
    /// Compute the effective block length for a given [`DType`].
    ///
    /// For fixed-width types where [`DType::element_size`] is known and large enough that
    /// `element_size * block_len_multiple` would exceed `block_size_target`, this reduces the
    /// block length so each block stays close to the target byte size.
    fn effective_block_len(&self, dtype: &DType) -> usize {
        let Some(block_size_target) = self.block_size_target else {
            return self.block_len_multiple;
        };
        match dtype.element_size() {
            Some(elem_size) if elem_size > 0 => {
                // `div_ceil` ensures we overshoot the block_size_target; therefore preventing
                // `write_stream` from combining adjacent 0.9 MiB chunks into one 1.8 MiB chunk.
                let max_rows = usize::try_from(block_size_target.div_ceil(elem_size as u64))
                    .unwrap_or(usize::MAX);
                self.block_len_multiple.min(max_rows).max(1)
            }
            _ => self.block_len_multiple,
        }
    }
}

/// Repartition a stream of arrays into blocks.
///
/// Each emitted block (except the last) is at least `block_size_minimum` bytes and contains a
/// multiple of `block_len_multiple` rows.
#[derive(Clone)]
pub struct RepartitionStrategy {
    child: Arc<dyn LayoutStrategy>,
    options: RepartitionWriterOptions,
    allowed_encodings: Arc<[ArrayId]>,
}

impl RepartitionStrategy {
    pub fn new<S: LayoutStrategy>(child: S, options: RepartitionWriterOptions) -> Self {
        Self {
            child: Arc::new(child),
            options,
            allowed_encodings: Arc::from([]),
        }
    }

    /// Keeps arrays of the given encodings, at any depth of structs and lists, instead of
    /// canonicalizing them.
    ///
    /// When [`Dict`] is allowed, a block made of dictionary chunks becomes a single dictionary
    /// holding only the distinct values its codes reference. A block made of one chunk of an
    /// allowed encoding is passed on as it is. Other blocks are canonicalized as usual.
    pub fn with_allowed_encodings(mut self, ids: impl IntoIterator<Item = ArrayId>) -> Self {
        self.allowed_encodings = ids.into_iter().collect();
        self
    }
}

#[async_trait]
impl LayoutStrategy for RepartitionStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        // TODO(os): spawn stream below like:
        // canon_stream = stream.map(async {to_canonical}).map(spawn).buffered(parallelism)
        let dtype = stream.dtype().clone();
        let allowed = Arc::clone(&self.allowed_encodings);
        let stream = if self.options.canonicalize {
            let canonicalize_session = session.clone();
            let allowed = Arc::clone(&allowed);
            SequentialStreamAdapter::new(
                dtype.clone(),
                stream.map(move |chunk| {
                    let (sequence_id, chunk) = chunk?;
                    if contains_allowed(&chunk, &allowed) {
                        return VortexResult::Ok((sequence_id, chunk));
                    }
                    let mut ctx = canonicalize_session.create_execution_ctx();
                    let canonical = chunk.execute::<Canonical>(&mut ctx)?.into_array();
                    VortexResult::Ok((sequence_id, canonical))
                }),
            )
            .sendable()
        } else {
            stream
        };

        let dtype_clone = dtype.clone();
        let options = self.options.clone();

        // For fixed-width types with large per-element sizes, reduce the block_len_multiple
        // so that each block targets block_size_target bytes rather than producing oversized
        // segments.
        let block_len = options.effective_block_len(&dtype);
        let block_size_minimum = options.block_size_minimum;
        let repartition_session = session.clone();

        let repartitioned_stream = try_stream! {
            let canonical_stream = stream.peekable();
            pin_mut!(canonical_stream);

            let mut ctx = repartition_session.create_execution_ctx();
            let mut chunks = ChunksBuffer::new(block_size_minimum, block_len);
            while let Some(chunk) = canonical_stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                let mut sequence_pointer = sequence_id.descend();
                let mut offset = 0;
                while offset < chunk.len() {
                    let end = (offset + block_len).min(chunk.len());
                    let sliced = chunk.slice(offset..end)?;
                    chunks.push_back(sliced, &mut ctx)?;
                    offset = end;

                    if chunks.have_enough() {
                        let output_chunks = chunks.collect_exact_blocks(&mut ctx)?;
                        assert!(!output_chunks.is_empty());
                        if let Some(block) =
                            finish_block(output_chunks, &dtype_clone, &allowed, &mut ctx)?
                        {
                            yield (sequence_pointer.advance(), block)
                        }
                    }
                }
                if canonical_stream.as_mut().peek().await.is_none() {
                    let to_flush = chunks.data.drain(..).map(|(arr, _)| arr).collect();
                    if let Some(block) =
                        finish_block(to_flush, &dtype_clone, &allowed, &mut ctx)?
                    {
                        yield (sequence_pointer.advance(), block)
                    }
                }
            }
        };

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, repartitioned_stream).sendable(),
                eof,
                session,
            )
            .await
    }
}

/// Buffers blocks until they hold enough rows and bytes to emit.
///
/// Bytes are the exact uncompressed size of each block, not the buffers it holds: a view array
/// read from Arrow points into whole decompressed pages, which `nbytes()` would count in full.
struct ChunksBuffer {
    /// Each entry stores the chunk and the size snapshot taken at push time.
    /// This avoids accounting mismatches when interior-mutable arrays (e.g. `SharedArray`)
    /// change their reported size after being pushed.
    data: VecDeque<(ArrayRef, u64)>,
    row_count: usize,
    nbytes: u64,
    block_size_minimum: u64,
    block_len_multiple: usize,
}

impl ChunksBuffer {
    fn new(block_size_minimum: u64, block_len_multiple: usize) -> Self {
        Self {
            data: Default::default(),
            row_count: 0,
            nbytes: 0,
            block_size_minimum,
            block_len_multiple,
        }
    }

    fn have_enough(&self) -> bool {
        self.nbytes >= self.block_size_minimum && self.row_count >= self.block_len_multiple
    }

    fn collect_exact_blocks(&mut self, ctx: &mut ExecutionCtx) -> VortexResult<Vec<ArrayRef>> {
        let nblocks = self.row_count / self.block_len_multiple;
        let mut res = Vec::with_capacity(self.data.len());
        let mut remaining = nblocks * self.block_len_multiple;
        while remaining > 0 {
            let (chunk, _) = self
                .pop_front()
                .vortex_expect("must have at least one chunk");
            let len = chunk.len();

            if len > remaining {
                let left = chunk.slice(0..remaining)?;
                let right = chunk.slice(remaining..len)?;
                self.push_front(right, ctx)?;
                res.push(left);
                remaining = 0;
            } else {
                res.push(chunk);
                remaining -= len;
            }
        }
        Ok(res)
    }

    fn push_back(&mut self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let nb = exact_size(&chunk, ctx)?;
        self.row_count += chunk.len();
        self.nbytes += nb;
        self.data.push_back((chunk, nb));
        Ok(())
    }

    fn push_front(&mut self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let nb = exact_size(&chunk, ctx)?;
        self.row_count += chunk.len();
        self.nbytes += nb;
        self.data.push_front((chunk, nb));
        Ok(())
    }

    fn pop_front(&mut self) -> Option<(ArrayRef, u64)> {
        let res = self.data.pop_front();
        if let Some((chunk, nb)) = res.as_ref() {
            self.row_count -= chunk.len();
            self.nbytes -= nb;
        }
        res
    }
}

/// Assembles the chunks of one block, or returns `None` for an empty block.
///
/// See [`RepartitionStrategy::with_allowed_encodings`] for which blocks keep their encodings.
fn finish_block(
    chunks: Vec<ArrayRef>,
    dtype: &DType,
    allowed: &[ArrayId],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let chunks: Vec<ArrayRef> = chunks.into_iter().filter(|c| !c.is_empty()).collect();
    if chunks.is_empty() {
        return Ok(None);
    }
    if chunks.iter().any(|chunk| contains_allowed(chunk, allowed)) {
        return concat_allowed(chunks, dtype, allowed, ctx).map(Some);
    }
    let chunked = ChunkedArray::try_new(chunks, dtype.clone())?;
    Ok(Some(
        chunked.into_array().execute::<Canonical>(ctx)?.into_array(),
    ))
}

/// Whether an array of an allowed encoding appears anywhere in `array`.
fn contains_allowed(array: &ArrayRef, allowed: &[ArrayId]) -> bool {
    !allowed.is_empty()
        && array
            .depth_first_traversal()
            .any(|child| allowed.contains(&child.encoding_id()))
}

/// Concatenates chunks, keeping the arrays of allowed encodings found at any depth of structs and
/// lists.
fn concat_allowed(
    chunks: Vec<ArrayRef>,
    dtype: &DType,
    allowed: &[ArrayId],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let chunks: Vec<ArrayRef> = chunks.into_iter().filter(|c| !c.is_empty()).collect();
    if chunks.is_empty() {
        return Ok(Canonical::empty(dtype).into_array());
    }
    if allowed.contains(&Dict.id())
        && let Some(dict) = concat_dicts(&chunks, dtype, ctx)?
    {
        return Ok(dict);
    }
    if let [chunk] = chunks.as_slice()
        && allowed.contains(&chunk.encoding_id())
    {
        return Ok(chunk.clone());
    }

    let len = chunks.iter().map(|c| c.len()).sum();
    match dtype {
        DType::Struct(fields, nullability) => {
            let structs = chunks
                .into_iter()
                .map(|chunk| chunk.execute::<StructArray>(ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            let validity = concat_validity(&structs, *nullability, ctx)?;
            let field_arrays = fields
                .fields()
                .enumerate()
                .map(|(idx, field_dtype)| {
                    let field_chunks = structs
                        .iter()
                        .map(|chunk| chunk.unmasked_field(idx).clone())
                        .collect();
                    concat_allowed(field_chunks, &field_dtype, allowed, ctx)
                })
                .collect::<VortexResult<Vec<_>>>()?;
            Ok(
                StructArray::try_new(fields.names().clone(), field_arrays, len, validity)?
                    .into_array(),
            )
        }
        DType::List(element_dtype, nullability) => {
            let lists = chunks
                .into_iter()
                .map(|chunk| match chunk.as_opt::<List>() {
                    Some(list) => Ok(list.into_owned()),
                    None => list_from_list_view(chunk.execute::<ListViewArray>(ctx)?, ctx),
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let validity = concat_validity(&lists, *nullability, ctx)?;

            let mut offsets = BufferMut::<u64>::with_capacity(len + 1);
            offsets.push(0);
            let mut elements = Vec::with_capacity(lists.len());
            for list in &lists {
                let list_offsets = list
                    .offsets()
                    .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
                    .execute::<PrimitiveArray>(ctx)?;
                let list_offsets = list_offsets.as_slice::<u64>();
                let (Some(&first), Some(&last)) = (list_offsets.first(), list_offsets.last())
                else {
                    continue;
                };
                let base = offsets.last().copied().unwrap_or_default();
                offsets.extend(list_offsets[1..].iter().map(|o| o - first + base));
                elements.push(
                    list.elements()
                        .slice(usize::try_from(first)?..usize::try_from(last)?)?,
                );
            }
            let elements = concat_allowed(elements, element_dtype, allowed, ctx)?;
            Ok(
                ListArray::try_new(elements, offsets.freeze().into_array(), validity, ctx)?
                    .into_array(),
            )
        }
        _ => Ok(ChunkedArray::try_new(chunks, dtype.clone())?
            .into_array()
            .execute::<Canonical>(ctx)?
            .into_array()),
    }
}

/// Concatenates the validity of `arrays`.
fn concat_validity<A: AsRef<ArrayRef>>(
    arrays: &[A],
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity> {
    if nullability == Nullability::NonNullable {
        return Ok(Validity::NonNullable);
    }
    let len = arrays.iter().map(|a| a.as_ref().len()).sum();
    let mut bits = BitBufferMut::with_capacity(len);
    for array in arrays {
        let array = array.as_ref();
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        bits.append_buffer(&mask.into_bit_buffer());
    }
    Ok(Validity::from_mask(bits.freeze().into(), nullability))
}

/// Concatenates dictionary chunks into one compact dictionary.
///
/// Each chunk's codes are rebased onto its values, then the values no code references are dropped
/// and duplicate values are merged. Only the values are hashed, never the rows, so chunks that
/// each carry a whole Parquet column chunk dictionary end up sharing one small dictionary.
///
/// A constant chunk, as slicing a dictionary with constant codes produces, contributes a single
/// value. Returns `None` unless every chunk is a dictionary or a constant and at least one is a
/// dictionary.
fn concat_dicts(
    chunks: &[ArrayRef],
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !chunks.iter().any(|c| c.is::<Dict>())
        || !chunks.iter().all(|c| c.is::<Dict>() || c.is::<Constant>())
    {
        return Ok(None);
    }

    let len = chunks.iter().map(|c| c.len()).sum();
    let codes_nullability = Nullability::from(chunks.iter().any(|c| {
        c.as_opt::<Dict>()
            .is_some_and(|dict| dict.codes().dtype().is_nullable())
    }));
    let codes_dtype = DType::Primitive(PType::U32, codes_nullability);
    let mut codes = BufferMut::<u32>::with_capacity(len);
    let mut validity = BitBufferMut::with_capacity(len);
    let mut values = Vec::with_capacity(chunks.len());
    let mut offset = 0u32;

    for chunk in chunks {
        let (chunk_values, chunk_codes) = match chunk.as_opt::<Dict>() {
            Some(dict) => (dict.values().clone(), Some(dict.codes().clone())),
            None => {
                let constant = chunk.as_::<Constant>();
                let value = ConstantArray::new(constant.scalar().clone(), 1).into_array();
                (value, None)
            }
        };
        let Some(next_offset) = u32::try_from(chunk_values.len())
            .ok()
            .and_then(|values_len| offset.checked_add(values_len))
        else {
            return Ok(None);
        };

        match chunk_codes {
            Some(chunk_codes) => {
                let chunk_codes = chunk_codes
                    .cast(codes_dtype.clone())?
                    .execute::<PrimitiveArray>(ctx)?;
                let mask = chunk_codes
                    .validity()?
                    .execute_mask(chunk_codes.len(), ctx)?;
                // Null slots may hold any code, so they are pointed at value 0 of the chunk.
                match mask.bit_buffer() {
                    AllOr::All => codes.extend(
                        chunk_codes
                            .as_slice::<u32>()
                            .iter()
                            .map(|code| code + offset),
                    ),
                    AllOr::None => codes.push_n(offset, chunk_codes.len()),
                    AllOr::Some(bits) => codes.extend(
                        chunk_codes
                            .as_slice::<u32>()
                            .iter()
                            .zip(bits.iter())
                            .map(|(code, valid)| if valid { code + offset } else { offset }),
                    ),
                }
                validity.append_buffer(&mask.into_bit_buffer());
            }
            None => {
                codes.push_n(offset, chunk.len());
                validity.append_n(true, chunk.len());
            }
        }

        values.push(chunk_values.cast(dtype.clone())?);
        offset = next_offset;
    }

    let values = ChunkedArray::try_new(values, dtype.clone())?
        .into_array()
        .execute::<Canonical>(ctx)?
        .into_array();
    let validity = validity.freeze();
    let (codes, values) = compact_dict_values(codes, &validity, values, ctx)?;

    let codes = PrimitiveArray::new(
        codes.freeze(),
        Validity::from_mask(validity.into(), codes_nullability),
    );
    // SAFETY: every code was rebased onto the values its chunk contributed, then remapped onto
    // the compacted values.
    let dict = unsafe { DictArray::new_unchecked(codes.into_array(), values) };
    Ok(Some(dict.into_array()))
}

/// Drops the values no valid code references and merges duplicate values, remapping the codes.
///
/// Values that cannot be hashed are only filtered, and values holding nulls are left as they are.
fn compact_dict_values(
    mut codes: BufferMut<u32>,
    validity: &BitBuffer,
    values: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(BufferMut<u32>, ArrayRef)> {
    let mut referenced = vec![false; values.len()];
    for (code, valid) in codes.iter().zip(validity.iter()) {
        if valid {
            referenced[*code as usize] = true;
        }
    }
    // Null codes point at value 0, so it is kept to stay in bounds.
    if let Some(first) = referenced.first_mut() {
        *first = true;
    }
    if values.all_invalid(ctx)? || !values.all_valid(ctx)? {
        return Ok((codes, values));
    }

    let mut remap = vec![0u32; values.len()];
    let mut kept = 0u32;
    for (slot, &is_referenced) in remap.iter_mut().zip(&referenced) {
        if is_referenced {
            *slot = kept;
            kept += 1;
        }
    }
    let values = values
        .filter(Mask::from(BitBuffer::from(referenced.clone())))?
        .execute::<Canonical>(ctx)?
        .into_array();

    let values = if values.is::<Primitive>() || values.is::<VarBinView>() {
        let unique = dict_encode(&values, ctx)?;
        let unique_codes = unique
            .codes()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(ctx)?;
        let unique_codes = unique_codes.as_slice::<u32>();
        for slot in &mut remap {
            *slot = unique_codes[*slot as usize];
        }
        unique.values().clone()
    } else {
        values
    };

    for code in codes.iter_mut() {
        *code = remap[*code as usize];
    }
    Ok((codes, values))
}

/// The bytes a block's rows reference, or the bytes it holds for a dtype that cannot be
/// measured, such as variant.
fn exact_size(chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    if !UncompressedSizeInBytes::supports_dtype(chunk.dtype()) {
        return Ok(chunk.nbytes());
    }
    let size = uncompressed_size_in_bytes(chunk, ctx)?;
    u64::try_from(size).map_err(|e| vortex_err!("block size does not fit in u64: {e}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::Struct;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::varbinview::BinaryView;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    use super::*;
    use crate::LayoutStrategy;
    use crate::arrays::shared::Shared;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    const ONE_MEG: u64 = 1 << 20;

    #[test]
    fn effective_block_len_variable_width() {
        // Utf8 has no known element_size, so block_len_multiple is unchanged.
        let dtype = DType::Utf8(NonNullable);
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 8192);
    }

    #[test]
    fn effective_block_len_very_large_elements() {
        // FixedSizeList(f64, 1_000_000) = 8_000_000 bytes/element.
        // 1 MiB / 8_000_000 = 0, clamped to max(1) = 1.
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F64, NonNullable)),
            1_000_000,
            NonNullable,
        );
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 1);
    }

    #[test]
    fn repartition_large_element_type_produces_small_blocks() -> VortexResult<()> {
        // Create a FixedSizeList(f64, 1000) array with 1000 lists.
        // Each list is 8000 bytes, so 1000 lists = 8 MiB total.
        // With block_size_target = 1 MiB, effective block_len = 132.
        // We expect the repartition to produce blocks of 132 rows each.
        let list_size: u32 = 1000;
        let num_lists: usize = 1000;
        let total_elements = list_size as usize * num_lists;

        let elements = PrimitiveArray::from_iter((0..total_elements).map(|i| i as f64));
        let fsl = FixedSizeListArray::new(
            elements.into_array(),
            list_size,
            Validity::NonNullable,
            num_lists,
        );

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let child = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let strategy = RepartitionStrategy::new(
            child,
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 8192,
                block_size_target: Some(ONE_MEG),
                canonicalize: false,
            },
        );

        let stream = fsl.into_array().to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        // The layout should be a ChunkedLayout with multiple children.
        // With 1000 rows and effective block_len = 132:
        //   - 7 full blocks of 132 rows = 924 rows
        //   - 1 remainder block of 76 rows
        //   - Total: 8 blocks, 1000 rows
        assert_eq!(layout.row_count(), num_lists as u64);

        // All non-last children should have 132 rows.
        let nchildren = layout.nchildren();
        assert!(nchildren > 1, "expected multiple chunks, got {nchildren}");

        for i in 0..nchildren - 1 {
            let child = layout.slot(i)?.vortex_expect("chunk slot present");
            assert_eq!(
                child.row_count(),
                132,
                "chunk {i} has {} rows, expected 132",
                child.row_count()
            );
        }

        // Last child gets the remainder.
        let last = layout
            .slot(nchildren - 1)?
            .vortex_expect("chunk slot present");
        assert_eq!(last.row_count(), 1000 - 132 * (nchildren as u64 - 1));

        Ok(())
    }

    #[test]
    fn repartition_small_element_type_unchanged() -> VortexResult<()> {
        // For f64 (8 bytes/element), effective block_len stays at 8192.
        // With 10000 elements and block_size_minimum=0, we get one block of 8192
        // and one remainder of 1808.
        let num_elements: usize = 10000;
        let elements = PrimitiveArray::from_iter((0..num_elements).map(|i| i as f64));

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let child = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let strategy = RepartitionStrategy::new(
            child,
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 8192,
                block_size_target: Some(ONE_MEG),
                canonicalize: false,
            },
        );

        let stream = elements.into_array().to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        assert_eq!(layout.row_count(), num_elements as u64);
        assert_eq!(layout.nchildren(), 2);
        assert_eq!(
            layout
                .slot(0)?
                .vortex_expect("chunk slot present")
                .row_count(),
            8192
        );
        assert_eq!(
            layout
                .slot(1)?
                .vortex_expect("chunk slot present")
                .row_count(),
            1808
        );

        Ok(())
    }

    /// View blocks sharing a buffer far larger than the bytes they reference must still coalesce
    /// up to `block_size_minimum`.
    #[test]
    fn repartition_coalesces_view_blocks_sharing_a_large_buffer() -> VortexResult<()> {
        const VALUE_LEN: usize = 27;
        const ROWS_PER_CHUNK: usize = 8192;
        const NUM_CHUNKS: usize = 10;

        // One 2 MiB buffer shared by every chunk, of which each row references 27 bytes.
        let page = ByteBuffer::from(vec![b'x'; 2 << 20]);
        let value = [b'x'; VALUE_LEN];
        let dtype = DType::Utf8(NonNullable);
        let chunks = (0..NUM_CHUNKS)
            .map(|_| {
                let views = (0..ROWS_PER_CHUNK)
                    .map(|row| {
                        BinaryView::make_view(&value, 0, u32::try_from(row * VALUE_LEN).unwrap())
                    })
                    .collect::<Buffer<BinaryView>>();
                // SAFETY: every view points at `VALUE_LEN` bytes inside the shared page.
                unsafe {
                    VarBinViewArray::new_unchecked(
                        views,
                        Arc::from([page.clone()]),
                        dtype.clone(),
                        Validity::NonNullable,
                    )
                }
                .into_array()
            })
            .collect::<Vec<_>>();
        let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = RepartitionStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            RepartitionWriterOptions {
                block_size_minimum: ONE_MEG,
                block_len_multiple: ROWS_PER_CHUNK,
                block_size_target: None,
                canonicalize: false,
            },
        );

        let stream = chunked.to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        // Each chunk references 8192 * (16 + 27) bytes, about 344 KiB, so three chunks make the
        // 1 MiB minimum: blocks of 3, 3, 3 chunks and a 1-chunk tail.
        assert_eq!(layout.row_count(), (NUM_CHUNKS * ROWS_PER_CHUNK) as u64);
        assert_eq!(layout.nchildren(), 4);
        for i in 0..3 {
            let child = layout.slot(i)?.vortex_expect("chunk slot present");
            assert_eq!(child.row_count(), (3 * ROWS_PER_CHUNK) as u64);
        }
        Ok(())
    }

    /// Regression test: `SharedArray` slices sharing an `Arc<Mutex<SharedState>>` can
    /// transition from Source to Cached when any one of them is canonicalized. This caused
    /// `pop_front` to panic with `attempt to subtract with overflow` because the buffer's
    /// running `nbytes` total was accumulated with the smaller Source-era values while
    /// `pop_front` subtracted the larger Cached-era values.
    #[test]
    fn chunks_buffer_pop_front_no_panic_after_shared_execution() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let n = 20_000usize;
        let block_len = 10_000usize;

        let constant = ConstantArray::new(42i64, n);
        let shared = Shared::new(constant.into_array());
        let shared_handle = shared.clone();
        let arr = shared.into_array();

        let s1 = arr.slice(0..block_len)?;
        let s2 = arr.slice(block_len..n)?;

        let mut buf = ChunksBuffer::new(0, block_len);
        buf.push_back(s1, &mut ctx)?;
        buf.push_back(s2, &mut ctx)?;

        let _output = buf.pop_front().unwrap();

        // Transition SharedState from Source to Cached for ALL slices sharing this Arc.
        use crate::arrays::shared::SharedArrayExt;
        let _canonical =
            shared_handle.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;

        // Before the fix this panicked with "attempt to subtract with overflow".
        let _s2 = buf.pop_front().unwrap();
        assert_eq!(buf.nbytes, 0);
        assert_eq!(buf.row_count, 0);

        Ok(())
    }

    fn str_dict(codes: &[u32], values: &[&str]) -> VortexResult<ArrayRef> {
        let codes = PrimitiveArray::from_iter(codes.iter().copied()).into_array();
        let values = VarBinViewArray::from_iter_str(values.iter().copied()).into_array();
        Ok(DictArray::try_new(codes, values)?.into_array())
    }

    #[test]
    fn concat_dicts_merges_and_compacts_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Utf8(NonNullable);
        let chunks = vec![
            str_dict(&[0, 1, 1], &["x", "y", "unused"])?,
            str_dict(&[1, 0], &["y", "x"])?,
            ConstantArray::new("x", 2).into_array(),
        ];
        let expected = ChunkedArray::try_new(chunks.clone(), dtype.clone())?;

        let block = finish_block(chunks, &dtype, &[Dict.id()], &mut ctx)?
            .vortex_expect("block is not empty");

        let dict = block
            .as_opt::<Dict>()
            .vortex_expect("block stays a dictionary");
        assert_eq!(dict.values().len(), 2);
        assert_arrays_eq!(&block, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn concat_preserves_dicts_nested_in_lists() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut list =
            |codes: &[u32], values: &[&str], offsets: Vec<u32>| -> VortexResult<ArrayRef> {
                let elements = StructArray::from_fields(&[("name", str_dict(codes, values)?)])?;
                Ok(ListArray::try_new(
                    elements.into_array(),
                    PrimitiveArray::from_iter(offsets).into_array(),
                    Validity::NonNullable,
                    &mut ctx,
                )?
                .into_array())
            };
        let first = list(&[0, 1, 2, 0], &["a", "b", "c"], vec![0, 1, 3, 4])?;
        // A slice whose offsets do not start at zero.
        let second = list(&[1, 1, 0], &["b", "d"], vec![0, 1, 3])?.slice(1..2)?;
        let dtype = first.dtype().clone();
        let chunks = vec![first, second];
        let expected = ChunkedArray::try_new(chunks.clone(), dtype.clone())?;

        let block = finish_block(chunks, &dtype, &[Dict.id()], &mut ctx)?
            .vortex_expect("block is not empty");

        let elements = block.as_::<List>().elements().clone();
        assert_eq!(elements.len(), 6);
        assert!(elements.as_::<Struct>().unmasked_field(0).is::<Dict>());
        assert_arrays_eq!(&block, &expected, &mut ctx);
        Ok(())
    }
}
