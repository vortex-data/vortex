// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::PATCH_CHUNK_SIZE;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpack_compress::bitpack_encode;

static PATCH_INDEX_BITPACK: AtomicBool = AtomicBool::new(false);

/// Toggles bitpacking of patch index children.
///
/// Off by default: on TPC-H shaped data the per-array patch sets are too small for FastLanes
/// packing to pay for itself, while the extra array node makes serialized trees larger and cold
/// file opens measurably slower. `VORTEX_PATCH_INDEX_BITPACK=1` or this toggle turns it on for
/// dense-patch workloads and for size measurements.
pub fn force_patch_index_bitpack(enabled: bool) {
    PATCH_INDEX_BITPACK.store(enabled, Ordering::Relaxed);
}

fn patch_index_bitpack() -> bool {
    static FROM_ENV: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("VORTEX_PATCH_INDEX_BITPACK").is_ok_and(|v| v == "1")
    });
    PATCH_INDEX_BITPACK.load(Ordering::Relaxed) || *FROM_ENV
}

/// Compresses the given patches by downscaling and bitpacking integers and checking for constant
/// values.
pub fn compress_patches(patches: Patches, ctx: &mut ExecutionCtx) -> VortexResult<Patches> {
    // Downscale and bitpack the patch indices.
    let indices = patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .narrow(ctx)?;
    let indices = bitpack_index_child(indices, ctx)?;

    // Check if the values are constant.
    let values = patches.values();
    let values = if values
        .statistics()
        .compute_is_constant(ctx)
        .unwrap_or_default()
    {
        ConstantArray::new(values.execute_scalar(0, ctx)?, values.len()).into_array()
    } else {
        values.clone()
    };
    let chunk_offsets = patches
        .chunk_offsets()
        .as_ref()
        .map(|offsets| {
            let offsets_primitive = offsets
                .clone()
                .execute::<PrimitiveArray>(ctx)?
                .narrow(ctx)?;
            bitpack_index_child(offsets_primitive, ctx)
        })
        .transpose()?;

    Patches::new(
        patches.array_len(),
        patches.offset(),
        indices,
        values,
        chunk_offsets,
    )
}

/// Bitpacks a non-nullable index child (patch indices or chunk offsets) at the exact bit width of
/// its maximum, so the packed form never needs patches of its own.
///
/// FastLanes packs in 1024-value chunks and pads the tail, so short children stay unpacked —
/// padding would outweigh the width saving.
fn bitpack_index_child(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if !patch_index_bitpack()
        || array.len() < PATCH_CHUNK_SIZE
        || array.dtype().is_nullable()
        || !array.ptype().is_unsigned_int()
    {
        return Ok(array.into_array());
    }
    let bit_width: u32 = match_each_unsigned_integer_ptype!(array.ptype(), |P| {
        let Some(max) = array.statistics().compute_max::<P>(ctx) else {
            return Ok(array.into_array());
        };
        if max == 0 {
            return Ok(array.into_array());
        }
        max.ilog2() + 1
    });
    let Ok(bit_width) = u8::try_from(bit_width) else {
        return Ok(array.into_array());
    };
    if usize::from(bit_width) >= array.ptype().bit_width() {
        return Ok(array.into_array());
    }
    Ok(bitpack_encode(&array, bit_width, None, ctx)?.into_array())
}
