// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for the frozen single-reference FoR wire format.

use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayParts;
use vortex_array::ArraySerialization;
use vortex_array::ArrayView;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use super::for_v1_id;
use crate::FoR;
use crate::FoRArray;
use crate::FoRData;
use crate::r#for::array::FoRArrayExt;

pub(super) fn serialize(array: ArrayView<'_, FoR>) -> ArraySerialization {
    // Note that we **only** serialize the optional scalar value (not including the dtype).
    let metadata = ScalarValue::to_proto_bytes(array.reference_scalar().value());
    ArraySerialization::from_array(for_v1_id(), array.array(), metadata)
}

pub(super) fn deserialize(
    parts: ArrayDeserialization<'_>,
    session: &VortexSession,
) -> VortexResult<FoRArray> {
    vortex_ensure!(
        parts.buffers.is_empty(),
        "FoRArray expects 0 buffers, got {}",
        parts.buffers.len()
    );
    if parts.children.len() != 1 {
        vortex_bail!(
            "Expected 1 child for FoR encoding, found {}",
            parts.children.len()
        )
    }

    let scalar_value = ScalarValue::from_proto_bytes(parts.metadata, parts.dtype, session)?;
    let reference = Scalar::try_new(parts.dtype.clone(), scalar_value)?;
    let encoded = parts.children.get(0, parts.dtype, parts.len)?;
    let slots = smallvec![Some(encoded)];

    let data = FoRData::try_new(reference)?;
    Array::try_from_parts(
        ArrayParts::new(FoR, parts.dtype.clone(), parts.len, data).with_slots(slots),
    )
}
