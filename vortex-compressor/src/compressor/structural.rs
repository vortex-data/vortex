// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recursive compression of structural arrays: lists, list views, and physical slots.

use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_error::VortexResult;

use super::ROOT_SCHEME_ID;
use crate::CascadingCompressor;
use crate::scheme::CompressorContext;

/// Child indices for the compressor's list/listview compression.
pub(super) mod root_list_children {
    /// List/ListView offsets child.
    pub const OFFSETS: usize = 1;
    /// ListView sizes child.
    pub const SIZES: usize = 2;
}

impl CascadingCompressor {
    /// Compresses a [`ListArray`] by narrowing offsets and recursively compressing elements.
    pub(super) fn compress_list_array(
        &self,
        list_array: ListArray,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let list_array = list_array.reset_offsets(true, exec_ctx)?;

        let compressed_elems = self.compress(list_array.elements(), exec_ctx)?;

        // Record the root scheme with the offsets child index so root exclusion rules apply.
        let offset_ctx =
            compress_ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let list_offsets_primitive = list_array
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_offsets_primitive),
            offset_ctx,
            exec_ctx,
        )?;

        Ok(
            ListArray::try_new(compressed_elems, compressed_offsets, list_array.validity()?)?
                .into_array(),
        )
    }

    /// Compresses a [`ListViewArray`] by narrowing offsets/sizes and recursively compressing
    /// elements.
    pub(super) fn compress_list_view_array(
        &self,
        list_view: ListViewArray,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compressed_elems = self.compress(list_view.elements(), exec_ctx)?;

        let offset_ctx = compress_ctx
            .clone()
            .descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let list_view_offsets_primitive = list_view
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_view_offsets_primitive),
            offset_ctx,
            exec_ctx,
        )?;

        let sizes_ctx = compress_ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::SIZES);
        let list_view_sizes_primitive = list_view
            .sizes()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_sizes = self.compress_canonical(
            Canonical::Primitive(list_view_sizes_primitive),
            sizes_ctx,
            exec_ctx,
        )?;

        Ok(ListViewArray::try_new(
            compressed_elems,
            compressed_offsets,
            compressed_sizes,
            list_view.validity()?,
        )?
        .into_array())
    }

    /// Compress very child slot of the array, then re-build it from them.
    pub(super) fn compress_physical_slots(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let slots = array
            .slots()
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .map(|child| self.compress(child, exec_ctx))
                    .transpose()
            })
            .collect::<VortexResult<ArraySlots>>()?;

        // SAFETY: compression rewrites each child slot to an equivalent physical representation,
        // preserving the parent array's logical values and statistics.
        unsafe { array.clone().with_slots(slots) }
    }
}
