// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.chunk_delta`, whose children are the deltas, the bases, the minimums and,
//! when there is one, the validity.

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ChunkDelta;
use crate::ChunkDeltaArrayExt;
use crate::ChunkDeltaArraySlotsExt;
use crate::FL_CHUNK_SIZE;

/// Metadata for `fastlanes.chunk_delta`.
#[derive(Clone, prost::Message)]
struct ChunkDeltaMetadata {
    /// The position of the first element within the first chunk.
    #[prost(uint32, tag = "1")]
    offset: u32,
    /// The byte width of the unsigned deltas.
    #[prost(uint32, tag = "2")]
    deltas_bytes: u32,
}

/// Serde for the [`ChunkDelta`] array.
///
/// Register this plugin, or call [`crate::initialize`], to enable serde.
#[derive(Clone, Debug)]
pub struct ChunkDeltaPlugin;

impl ArrayPlugin for ChunkDeltaPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&ChunkDelta)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<ChunkDelta>().ok_or_else(|| {
            vortex_err!("ChunkDelta plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = ChunkDeltaMetadata {
            offset: u32::from(view.offset()),
            deltas_bytes: u32::try_from(view.deltas().dtype().as_ptype().byte_width())?,
        };
        Ok(Some(ArraySerialization::from_array(
            self.id(),
            array,
            metadata.encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            parts.buffers.is_empty(),
            "ChunkDelta expects 0 buffers, got {}",
            parts.buffers.len()
        );
        let metadata = ChunkDeltaMetadata::decode(parts.metadata)?;
        let offset = u16::try_from(metadata.offset)?;
        vortex_ensure!(
            usize::from(offset) < FL_CHUNK_SIZE,
            "ChunkDelta offset must be less than {FL_CHUNK_SIZE}, got {offset}"
        );
        let deltas_ptype = match metadata.deltas_bytes {
            1 => PType::U8,
            2 => PType::U16,
            4 => PType::U32,
            8 => PType::U64,
            other => vortex_bail!("ChunkDelta deltas cannot be {other} bytes wide"),
        };
        let span = usize::from(offset) + parts.len;
        let deltas = parts.children.get(
            0,
            &DType::Primitive(deltas_ptype, Nullability::NonNullable),
            span,
        )?;
        let chunks = span.div_ceil(FL_CHUNK_SIZE);
        let bases = parts.children.get(1, &parts.dtype.as_nonnullable(), chunks)?;
        let mins = parts.children.get(2, &parts.dtype.as_nonnullable(), chunks)?;
        let validity = match parts.children.len() {
            3 => Validity::from(parts.dtype.nullability()),
            4 => Validity::Array(parts.children.get(3, &Validity::DTYPE, parts.len)?),
            n => vortex_bail!("ChunkDelta expects 3 or 4 children, got {n}"),
        };
        Ok(ChunkDelta::try_new(deltas, bases, mins, validity, parts.len, offset)?.into_array())
    }
}
