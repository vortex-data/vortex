// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire translation for the two DBP formats.
//!
//! The plugin restores the [MSP contract](crate) on read and writes the stored integer child.
//! Each version owns its metadata schema.

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DecimalByteParts;
use super::DecimalBytePartsArraySlotsExt;

mod v1;
mod v2;

pub use v2::DecimalBytePartsV2Metadata;

/// The frozen single-child DBP serialized ID.
pub fn decimal_byte_parts_v1_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.decimal_byte_parts");
    *ID
}

/// The current in-memory DBP ID and serialized ID for arrays with lower parts.
pub fn decimal_byte_parts_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.decimal_byte_parts.v2");
    *ID
}

/// Serde for the [`DecimalByteParts`] array using the frozen v1 and v2 wire formats.
///
/// Each version owns its metadata schema and serde functions. The plugin writes v1 whenever an
/// array has no lower parts, so such arrays stay readable by older readers, and v2 otherwise.
/// The v2 format itself accepts any lower part count up to the maximum.
///
/// Register this plugin, or call [`crate::initialize`], to enable both formats.
/// [`DecimalByteParts`] does not implement [`ArrayPlugin`], so it cannot be registered on its own.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsPlugin;

impl ArrayPlugin for DecimalBytePartsPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&DecimalByteParts)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<DecimalByteParts>().ok_or_else(|| {
            vortex_err!(
                "DecimalByteParts plugin cannot serialize {}",
                array.encoding_id()
            )
        })?;
        let serialized = if view.lower_parts().is_empty() {
            v1::serialize(view)?
        } else {
            v2::serialize(view)?
        };
        Ok(Some(serialized))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let array = if parts.serialized_id == decimal_byte_parts_v1_id() {
            v1::deserialize(parts, session)?
        } else if parts.serialized_id == decimal_byte_parts_v2_id() {
            v2::deserialize(parts, session)?
        } else {
            vortex_bail!(
                "DecimalByteParts plugin does not recognize serialized ID {}",
                parts.serialized_id
            )
        };
        Ok(array.into_array())
    }
}

/// The wire stores the narrow integer child, so older readers need no `Widen` support.
fn stored_msp(msp: &ArrayRef) -> ArrayRef {
    match msp.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => msp.clone(),
    }
}

/// Restore the logical MSP dtype from a stored signed integer width.
///
/// Only wider legacy MSPs need a checked copy. Their null payloads may be outside the target
/// range, so the primitive cast kernel masks failures and zeroes null lanes.
fn read_msp(
    msp: ArrayRef,
    decimal_dtype: DecimalDType,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let logical_ptype = super::msp_ptype(decimal_dtype);
    let stored_ptype = msp.dtype().as_ptype();
    let logical_dtype = DType::Primitive(logical_ptype, msp.dtype().nullability());

    if stored_ptype.byte_width() == logical_ptype.byte_width() {
        return Ok(msp);
    }

    if stored_ptype.byte_width() < logical_ptype.byte_width() {
        return Ok(WidenArray::try_new(msp, logical_dtype)?.into_array());
    }

    vortex_ensure!(
        msp.is_host(),
        "a wider legacy DecimalByteParts MSP must be in host memory, got device memory"
    );
    let mut ctx = session.create_execution_ctx();
    let msp = msp.execute::<PrimitiveArray>(&mut ctx)?;
    vortex_ensure!(
        msp.buffer_handle().is_on_host(),
        "a wider legacy DecimalByteParts MSP must be in host memory, got device memory"
    );
    Ok(msp
        .into_array()
        .cast(logical_dtype)?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array())
}

#[cfg(test)]
mod tests;
