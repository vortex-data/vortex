// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recursive compression of structural arrays: lists, list views, and physical slots.

use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolSlots;
use vortex_array::arrays::decimal::DecimalSlots;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::primitive::PrimitiveSlots;
use vortex_array::arrays::varbinview::VarBinViewSlots;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

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
    /// Compresses a validity bitmap with the enabled boolean schemes, returning `None` when the
    /// result's buffers are no smaller than the bitmap's. The comparison is of buffer bytes, so it
    /// does not count the metadata a compressed bitmap adds when serialized.
    fn compress_bitmap(
        &self,
        bitmap: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let compressed = self.compress(bitmap, exec_ctx)?;
        Ok((compressed.nbytes() < bitmap.nbytes()).then_some(compressed))
    }

    /// Compresses an array-backed validity mask using the enabled boolean schemes.
    ///
    /// # Errors
    ///
    /// Returns an error if compression of the validity array fails.
    pub(super) fn compress_validity(
        &self,
        validity: Validity,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Validity> {
        match validity {
            Validity::Array(bitmap) => Ok(Validity::Array(
                self.compress_bitmap(&bitmap, exec_ctx)?.unwrap_or(bitmap),
            )),
            validity => Ok(validity),
        }
    }

    /// Compresses the validity bitmap of a bool, primitive, decimal or string leaf in place,
    /// leaving the leaf unchanged when the bitmap does not get smaller.
    ///
    /// # Errors
    ///
    /// Returns an error if compression of the validity array fails.
    pub(super) fn compress_leaf_validity(
        &self,
        canonical: Canonical,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Canonical> {
        let (slot, array): (usize, &ArrayRef) = match &canonical {
            Canonical::Bool(array) => (BoolSlots::VALIDITY_CHILD, array.as_ref()),
            Canonical::Primitive(array) => (PrimitiveSlots::VALIDITY_CHILD, array.as_ref()),
            Canonical::Decimal(array) => (DecimalSlots::VALIDITY_CHILD, array.as_ref()),
            Canonical::VarBinView(array) => (VarBinViewSlots::VALIDITY_CHILD, array.as_ref()),
            _ => return Ok(canonical),
        };
        let Some(validity) = &array.slots()[slot] else {
            return Ok(canonical);
        };
        let Some(compressed) = self.compress_bitmap(validity, exec_ctx)? else {
            return Ok(canonical);
        };
        let mut slots: ArraySlots = array.slots().iter().cloned().collect();
        slots[slot] = Some(compressed);
        // SAFETY: the validity slot is replaced with an equivalent non-nullable boolean array
        // of the same length. The leaf's values, dtype, and other slots are unchanged.
        unsafe { array.clone().with_slots(slots) }?.execute::<Canonical>(exec_ctx)
    }

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

        Ok(ListArray::try_new(
            compressed_elems,
            compressed_offsets,
            self.compress_validity(list_array.validity()?, exec_ctx)?,
            exec_ctx,
        )?
        .into_array())
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
            self.compress_validity(list_view.validity()?, exec_ctx)?,
            exec_ctx,
        )?
        .into_array())
    }

    /// Compresses a [`MapArray`] by recursively compressing its entries [`ListViewArray`].
    pub(super) fn compress_map_array(
        &self,
        map_array: MapArray,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let data_parts = map_array.into_data_parts();
        let entries = self.compress_list_view_array(data_parts.entries, compress_ctx, exec_ctx)?;
        let entries = entries
            .as_opt::<ListView>()
            .ok_or_else(|| vortex_err!("Compressed map entries became {}", entries.encoding_id()))?
            .into_owned();

        Ok(MapArray::try_new(data_parts.map_dtype, entries)?.into_array())
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
