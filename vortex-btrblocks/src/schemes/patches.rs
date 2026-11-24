// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::patched::PatchedSlots;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesParts;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::Child;
use crate::EncodedChildren;
use crate::Scheme;

/// Compresses the given patches by downscaling integers and checking for constant values.
pub fn compress_patches(patches: Patches, ctx: &mut ExecutionCtx) -> VortexResult<Patches> {
    // Downscale the patch indices.
    let indices = patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .narrow(ctx)?
        .into_array();

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
                .narrow(ctx)?
                .into_array();
            Ok::<ArrayRef, VortexError>(offsets_primitive)
        })
        .transpose()?;

    Patches::new(
        patches.array_len(),
        patches.offset(),
        indices,
        values,
        chunk_offsets,
        ctx,
    )
}

/// Patches a scheme stores as they are, registered as kept children.
///
/// Patches hold the few values an encoding cannot represent, so they are not compressed further.
pub(crate) enum KeptPatches {
    /// The encoding has no patches.
    None,
    /// Patches compressed by [`compress_patches`] and stored inside the encoding.
    Interior {
        /// The length of the patched array.
        array_len: usize,
        /// The offset of the patches into the patched array.
        offset: usize,
        /// Whether the patches have chunk offsets.
        has_chunk_offsets: bool,
        /// The offset of the patches into their first chunk.
        offset_within_chunk: Option<usize>,
    },
    /// Patches in a [`Patched`] array that wraps the encoding, which `use_experimental_patches`
    /// selects. The array holds a placeholder for the encoding until it is built.
    Patched(ArrayRef),
}

impl KeptPatches {
    /// Returns the arrays of `patches` as kept children of `scheme`, numbered from `index`.
    ///
    /// The patches go in a [`Patched`] array around the encoding when `use_experimental_patches`
    /// is set, and inside the encoding otherwise. `array` is the input that `patches` patch.
    ///
    /// # Errors
    ///
    /// Returns an error if the patches cannot be narrowed or transposed.
    pub(crate) fn new(
        scheme: &dyn Scheme,
        index: usize,
        array: &ArrayRef,
        patches: Option<Patches>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(Self, Vec<Child>)> {
        let Some(patches) = patches else {
            return Ok((Self::None, Vec::new()));
        };

        if use_experimental_patches() {
            let patched = Patched::from_array_and_patches(array.clone(), &patches, ctx)?;
            let keep = vec![
                Child::new(scheme, index, patched.lane_offsets().clone()),
                Child::new(scheme, index + 1, patched.patch_indices().clone()),
                Child::new(scheme, index + 2, patched.patch_values().clone()),
            ];
            return Ok((Self::Patched(patched.into_array()), keep));
        }

        Self::interior(scheme, index, Some(patches), ctx)
    }

    /// Returns the arrays of `patches`, stored inside the encoding however
    /// `use_experimental_patches` is set, as kept children of `scheme`, numbered from `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if the patches cannot be narrowed.
    pub(crate) fn interior(
        scheme: &dyn Scheme,
        index: usize,
        patches: Option<Patches>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(Self, Vec<Child>)> {
        let Some(patches) = patches else {
            return Ok((Self::None, Vec::new()));
        };
        let PatchesParts {
            array_len,
            offset,
            indices,
            values,
            chunk_offsets,
            offset_within_chunk,
        } = compress_patches(patches, ctx)?.into_parts();
        let has_chunk_offsets = chunk_offsets.is_some();
        let mut keep = Vec::with_capacity(2 + usize::from(has_chunk_offsets));
        keep.push(Child::new(scheme, index, indices));
        keep.push(Child::new(scheme, index + 1, values));
        if let Some(chunk_offsets) = chunk_offsets {
            keep.push(Child::new(scheme, index + 2, chunk_offsets));
        }
        let patches = Self::Interior {
            array_len,
            offset,
            has_chunk_offsets,
            offset_within_chunk,
        };
        Ok((patches, keep))
    }

    /// Builds the encoding with `build`, which receives the patches to store inside it, then wraps
    /// it in its [`Patched`] array, if it has one.
    ///
    /// # Errors
    ///
    /// Returns an error if a kept child is missing, `build` fails, or the encoding it builds does
    /// not match the placeholder in the [`Patched`] array.
    pub(crate) fn build(
        self,
        children: &mut EncodedChildren,
        build: impl FnOnce(Option<Patches>) -> VortexResult<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        match self {
            Self::None => build(None),
            Self::Interior {
                array_len,
                offset,
                has_chunk_offsets,
                offset_within_chunk,
            } => {
                let [indices, values] = children.kept()?;
                let chunk_offsets = if has_chunk_offsets {
                    let [chunk_offsets] = children.kept()?;
                    Some(chunk_offsets)
                } else {
                    None
                };
                // SAFETY: these are the parts of valid patches, with their arrays kept unchanged.
                let patches = unsafe {
                    Patches::new_unchecked(
                        array_len,
                        offset,
                        indices,
                        values,
                        chunk_offsets,
                        offset_within_chunk,
                    )
                };
                build(Some(patches))
            }
            Self::Patched(patched) => {
                let inner = build(None)?;
                let [lane_offsets, patch_indices, patch_values] = children.kept()?;
                let slots = PatchedSlots {
                    inner,
                    lane_offsets,
                    patch_indices,
                    patch_values,
                }
                .into_slots();
                // SAFETY: the encoding replaces the placeholder, its input, which it matches except
                // where the patches override it, so the patched values are unchanged; the array
                // has no statistics yet. The patches' arrays were kept as they were.
                unsafe { patched.with_slots(slots) }
            }
        }
    }
}
