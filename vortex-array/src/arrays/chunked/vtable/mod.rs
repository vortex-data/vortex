// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::arrays::ChunkedArray;
use crate::serde::ArrayChildren;
use crate::vtable::{NotSupported, VTable};
use crate::{EmptyMetadata, EncodingId, EncodingRef, ToCanonical, vtable};

mod array;
mod canonical;
mod compute;
mod operations;
mod validity;
mod visitor;

vtable!(Chunked);

impl VTable for ChunkedVTable {
    type Array = ChunkedArray;
    type Encoding = ChunkedEncoding;
    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = NotSupported;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.chunked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ChunkedEncoding.as_ref())
    }

    fn metadata(_array: &ChunkedArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        _encoding: &ChunkedEncoding,
        dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ChunkedArray> {
        if children.is_empty() {
            vortex_bail!("Chunked array needs at least one child");
        }

        let nchunks = children.len() - 1;

        // The first child contains the row offsets of the chunks
        let chunk_offsets = children
            .get(
                0,
                &DType::Primitive(PType::U64, Nullability::NonNullable),
                // 1 extra offset for the end of the last chunk
                nchunks + 1,
            )?
            .to_primitive()
            .buffer::<u64>();

        // The remaining children contain the actual data of the chunks
        let chunks = chunk_offsets
            .iter()
            .tuple_windows()
            .enumerate()
            .map(|(idx, (start, end))| {
                let chunk_len = usize::try_from(end - start)
                    .map_err(|_| vortex_err!("chunk_len {} exceeds usize range", end - start))?;
                children.get(idx + 1, dtype, chunk_len)
            })
            .try_collect()?;

        // SAFETY: All chunks are deserialized with the same dtype that was serialized.
        // Each chunk was validated during deserialization to match the expected dtype.
        unsafe { Ok(ChunkedArray::new_unchecked(chunks, dtype.clone())) }
    }
}

#[derive(Clone, Debug)]
pub struct ChunkedEncoding;
