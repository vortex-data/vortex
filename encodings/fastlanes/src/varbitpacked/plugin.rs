// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.varbitpacked`, which stores the packed chunks and their widths as buffers
//! and the validity, when there is one, as its only child.

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::VarBitPacked;
use crate::VarBitPackedArrayExt;

/// Metadata for `fastlanes.varbitpacked`.
#[derive(Clone, prost::Message)]
struct VarBitPackedMetadata {
    /// The position of the first element within the first chunk.
    #[prost(uint32, tag = "1")]
    offset: u32,
}

/// Serde for the [`VarBitPacked`] array.
///
/// Register this plugin, or call [`crate::initialize`], to enable serde.
#[derive(Clone, Debug)]
pub struct VarBitPackedPlugin;

impl ArrayPlugin for VarBitPackedPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&VarBitPacked)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<VarBitPacked>().ok_or_else(|| {
            vortex_err!("VarBitPacked plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = VarBitPackedMetadata {
            offset: u32::from(view.offset()),
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
            parts.buffers.len() == 2,
            "VarBitPacked expects 2 buffers, got {}",
            parts.buffers.len()
        );
        let metadata = VarBitPackedMetadata::decode(parts.metadata)?;
        let validity = match parts.children.len() {
            0 => Validity::from(parts.dtype.nullability()),
            1 => Validity::Array(parts.children.get(0, &Validity::DTYPE, parts.len)?),
            n => vortex_bail!("VarBitPacked expects 0 or 1 children, got {n}"),
        };
        Ok(VarBitPacked::try_new(
            parts.buffers[0].clone(),
            parts.buffers[1].clone(),
            parts.dtype.as_ptype(),
            validity,
            parts.len,
            u16::try_from(metadata.offset)?,
        )?
        .into_array())
    }
}
