// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.affine`, which stores the residuals and the three per-chunk parameter
//! arrays as children.

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Affine;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::MAX_SLOPE_SHIFT;
use crate::affine::array::num_chunks;

/// The serialized and in-memory ID of the [`Affine`] array.
pub fn affine_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.affine");
    *ID
}

/// Metadata for `fastlanes.affine`. The children are encoded, references, scales and slopes.
#[derive(Clone, prost::Message)]
struct AffineMetadata {
    /// The position of the first element within the first chunk.
    #[prost(uint32, tag = "1")]
    offset: u32,
    /// The number of fractional bits in each slope.
    #[prost(uint32, tag = "2")]
    slope_shift: u32,
}

/// Serde for the [`Affine`] array.
///
/// Register this plugin, or call [`crate::initialize`], to enable serde.
#[derive(Clone, Debug)]
pub struct AffinePlugin;

impl ArrayPlugin for AffinePlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&Affine)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<Affine>().ok_or_else(|| {
            vortex_err!("Affine plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = AffineMetadata {
            offset: u32::from(view.offset()),
            slope_shift: u32::from(view.slope_shift()),
        };
        Ok(Some(ArraySerialization::new(
            affine_id(),
            metadata.encode_to_vec(),
            vec![],
            vec![
                view.encoded().clone(),
                view.references().clone(),
                view.scales().clone(),
                view.slopes().clone(),
            ],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            parts.buffers.is_empty(),
            "AffineArray expects 0 buffers, got {}",
            parts.buffers.len()
        );
        vortex_ensure!(
            parts.children.len() == 4,
            "Expected 4 children for {}, found {}",
            affine_id(),
            parts.children.len()
        );
        let metadata = AffineMetadata::decode(parts.metadata)?;
        vortex_ensure!(
            usize::try_from(metadata.offset).is_ok_and(|offset| offset < FL_CHUNK_SIZE),
            "Affine offset must be less than {FL_CHUNK_SIZE}, got {}",
            metadata.offset
        );
        vortex_ensure!(
            metadata.slope_shift <= u32::from(MAX_SLOPE_SHIFT),
            "Affine slope shift must be at most {MAX_SLOPE_SHIFT}, got {}",
            metadata.slope_shift
        );
        let offset = u16::try_from(metadata.offset)?;
        let chunks = num_chunks(offset, parts.len);
        let param_dtype = parts.dtype.as_nonnullable();
        let encoded = parts.children.get(0, parts.dtype, parts.len)?;
        let references = parts.children.get(1, &param_dtype, chunks)?;
        let scales = parts.children.get(2, &param_dtype, chunks)?;
        let slopes = parts
            .children
            .get(3, &DType::Primitive(PType::I64, false.into()), chunks)?;
        Ok(Affine::try_new(
            encoded,
            references,
            scales,
            slopes,
            offset,
            u8::try_from(metadata.slope_shift)?,
        )?
        .into_array())
    }
}
