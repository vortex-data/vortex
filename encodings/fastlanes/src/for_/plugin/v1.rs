// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.for`, which stores a single reference in its metadata.

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use super::for_v1_id;
use crate::FoR;
use crate::for_::array::FoRArraySlotsExt;

pub(super) fn serialize(array: ArrayView<'_, FoR>, reference: &Scalar) -> ArraySerialization {
    // Note that we **only** serialize the optional scalar value (not including the dtype).
    let metadata = ScalarValue::to_proto_bytes(reference.value());
    ArraySerialization::new(for_v1_id(), metadata, vec![], vec![array.encoded().clone()])
}

pub(super) fn deserialize(
    parts: ArrayDeserialization<'_>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
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
    Ok(FoR::try_new(encoded, reference)?.into_array())
}
