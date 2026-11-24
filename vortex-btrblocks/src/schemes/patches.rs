// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
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
/// Patches hold the rare values an encoding cannot represent, so they are not compressed.
pub(crate) enum KeptPatches {
    /// The encoding has no patches.
    None,
    /// Patches narrowed by [`compress_patches`] and stored inside the encoding.
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
    /// Registers the arrays of `patches` as kept children of `scheme`, numbered from `index`.
    ///
    /// `array` is the input that `patches` patch.
    ///
    /// # Errors
    ///
    /// Returns an error if the patches cannot be narrowed or transposed.
    pub(crate) fn new(
        scheme: &dyn Scheme,
        index: usize,
        array: &ArrayRef,
        patches: Option<Patches>,
        keep: &mut Vec<Child>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let Some(patches) = patches else {
            return Ok(Self::None);
        };

        if use_experimental_patches() {
            let patched = Patched::from_array_and_patches(array.clone(), &patches, ctx)?;
            keep.extend(
                [
                    patched.lane_offsets(),
                    patched.patch_indices(),
                    patched.patch_values(),
                ]
                .into_iter()
                .zip(index..)
                .map(|(child, index)| Child::new(scheme, index, child.clone())),
            );
            return Ok(Self::Patched(patched.into_array()));
        }

        let PatchesParts {
            array_len,
            offset,
            indices,
            values,
            chunk_offsets,
            offset_within_chunk,
        } = compress_patches(patches, ctx)?.into_parts();
        let has_chunk_offsets = chunk_offsets.is_some();
        keep.extend(
            [Some(indices), Some(values), chunk_offsets]
                .into_iter()
                .flatten()
                .zip(index..)
                .map(|(child, index)| Child::new(scheme, index, child)),
        );
        Ok(Self::Interior {
            array_len,
            offset,
            has_chunk_offsets,
            offset_within_chunk,
        })
    }

    /// Builds the encoding with `build`, which receives the patches to store inside it, then wraps
    /// it in its [`Patched`] array, if it has one.
    ///
    /// # Errors
    ///
    /// Returns an error if a kept child is missing, or `build` fails.
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
                // SAFETY: the arrays are those of patches that were valid, kept as they were.
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
                let [lane_offsets, indices, values] = children.kept()?;
                // SAFETY: the placeholder is replaced with the encoding of the same values, and
                // the patches' arrays are kept as they were.
                unsafe {
                    patched.with_slots(
                        [inner, lane_offsets, indices, values]
                            .into_iter()
                            .map(Some)
                            .collect(),
                    )
                }
            }
        }
    }
}
