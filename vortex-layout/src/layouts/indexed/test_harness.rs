// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checks for an index kind's serialization, independent of writing a layout.

use vortex_array::ExecutionCtx;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::layouts::indexed::IndexVTable;

/// Check that `chunk` and `options` survive `vtable`'s serialization for a column of
/// `data_dtype`.
///
/// The chunk must encode as the kind's declared index dtype, and re-encoding its decoding must
/// reproduce that encoding exactly. The options must likewise re-serialize to the same blob once
/// parsed.
pub fn check_roundtrip<V: IndexVTable>(
    vtable: &V,
    data_dtype: &DType,
    chunk: &V::Chunk,
    options: &V::Options,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let blob = vtable.serialize_options(options);
    let parsed = vtable.deserialize_options(&blob)?;
    vortex_ensure!(
        vtable.serialize_options(&parsed) == blob,
        "Index {} options do not re-serialize to the same blob",
        vtable.id()
    );

    let declared = vtable
        .index_dtype(data_dtype, options)
        .ok_or_else(|| vortex_err!("Index {} cannot index {data_dtype}", vtable.id()))?;
    let encoded = vtable.encode(chunk, options, ctx)?;
    vortex_ensure!(
        encoded.dtype() == &declared,
        "Index {} encoded dtype {}, but declared {declared}",
        vtable.id(),
        encoded.dtype()
    );

    let decoded = vtable.decode(encoded.clone(), &parsed, ctx)?;
    let reencoded = vtable.encode(&decoded, &parsed, ctx)?;
    assert_arrays_eq!(reencoded, encoded, ctx);
    Ok(())
}
