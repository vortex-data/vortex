// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;

use crate::buffer::BufferHandle;
use crate::dtype::Nullability;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

pub(crate) fn buffer(array_name: &str, values: &BufferHandle, idx: usize) -> BufferHandle {
    match idx {
        0 => values.clone(),
        _ => vortex_panic!("{array_name} buffer index {idx} out of bounds"),
    }
}

pub(crate) fn buffer_name(idx: usize) -> Option<String> {
    match idx {
        0 => Some("values".to_string()),
        _ => None,
    }
}

pub(crate) fn single_buffer(buffers: &[BufferHandle]) -> VortexResult<BufferHandle> {
    vortex_ensure_eq!(buffers.len(), 1);
    Ok(buffers[0].clone())
}

/// Returns the validity stored in `children`, or the validity that `nullability` implies when
/// there is no child.
///
/// # Errors
///
/// Returns an error if there is more than one child, or if a non-nullable node has a validity
/// child.
pub(crate) fn deserialize_validity(
    nullability: Nullability,
    len: usize,
    children: &dyn ArrayChildren,
) -> VortexResult<Validity> {
    match children.len() {
        0 => Ok(Validity::from(nullability)),
        1 => {
            // A non-nullable node never has a validity child. Accepting one would build an array
            // whose validity does not match its dtype.
            vortex_ensure!(
                nullability.is_nullable(),
                "A validity child requires a nullable dtype, got a non-nullable dtype"
            );

            Ok(Validity::Array(children.get(0, &Validity::DTYPE, len)?))
        }
        child_count => vortex_bail!("Expected 0 or 1 child, got {child_count}"),
    }
}
