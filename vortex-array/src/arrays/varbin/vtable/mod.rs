// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbin::VarBinData;
use crate::arrays::varbin::VarBinSlots;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_varbin_builder;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
pub(crate) mod canonical;
mod kernel;
mod operations;
mod validity;

use canonical::varbin_decode_views;
use canonical::varbin_to_canonical;
use vortex_session::VortexSession;

use crate::EqMode;
use crate::arrays::varbin::compute::rules::PARENT_RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;

/// A [`VarBin`]-encoded Vortex array.
pub type VarBinArray = Array<VarBin>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

#[derive(Clone, prost::Message)]
pub struct VarBinMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) offsets_ptype: i32,
}

impl ArrayHash for VarBinData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.bytes().array_hash(state, accuracy);
    }
}

impl ArrayEq for VarBinData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.bytes().array_eq(other.bytes(), accuracy)
    }
}

impl VTable for VarBin {
    type TypedArrayData = VarBinData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.varbin");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn validate(
        &self,
        _data: &VarBinData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == VarBinSlots::COUNT,
            "VarBinArray expected {} slots, found {}",
            VarBinSlots::COUNT,
            slots.len()
        );
        let offsets = slots[VarBinSlots::OFFSETS]
            .as_ref()
            .vortex_expect("VarBinArray offsets slot");
        vortex_ensure!(
            offsets.len().saturating_sub(1) == len,
            "VarBinArray length {} does not match outer length {}",
            offsets.len().saturating_sub(1),
            len
        );
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            "VarBinArray dtype must be binary or utf8, got {dtype}"
        );
        Ok(())
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bytes_handle().clone(),
            _ => vortex_panic!("VarBinArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("bytes".to_string()),
            _ => vortex_panic!("VarBinArray buffer_name index {idx} out of bounds"),
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );
        let mut data = array.data().clone();
        data.bytes = buffers[0].clone();
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            VarBinMetadata {
                offsets_ptype: PType::try_from(array.offsets().dtype())
                    .vortex_expect("Must be a valid PType") as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = VarBinMetadata::decode(metadata)?;
        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", children.len());
        };

        let offsets = children.get(
            0,
            &DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let bytes = buffers[0].clone().try_to_host_sync()?;

        let data = VarBinData::try_build(offsets.clone(), bytes, dtype.clone(), validity.clone())?;
        let slots = VarBinData::make_slots(offsets, &validity, len);
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        VarBinSlots::NAMES[idx].to_string()
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if let Some(result) =
            match_each_varbin_builder!(builder, |builder| builder.append_varbin(array, ctx))
        {
            return result;
        }

        // The two arms here are every builder a `Utf8`/`Binary` dtype has: all four
        // `VarBinBuilder` widths above, and `VarBinViewBuilder` below.
        let Some(view_builder) = builder.as_any().downcast_ref::<VarBinViewBuilder>() else {
            vortex_bail!("append_to_builder for VarBin requires a variable-binary builder")
        };

        if view_builder.compacts_buffers() {
            // A compacting builder decides per buffer whether to keep, slice or rewrite it, which
            // it can only do by measuring the finished views against the buffer. Go through the
            // canonical array so that policy still applies.
            return varbin_to_canonical(array, ctx)?
                .into_array()
                .append_to_builder(builder, ctx);
        }

        let builder = builder
            .as_any_mut()
            .downcast_mut::<VarBinViewBuilder>()
            .vortex_expect("builder type checked above");
        append_to_varbinview(array, builder, ctx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            varbin_to_canonical(array.as_view(), ctx)?.into_array(),
        ))
    }
}

/// Hands the value bytes to `builder` as a data buffer with views built over them.
///
/// Canonicalizing first would build the same views, then pay for them twice more: once to wrap
/// them in a `VarBinViewArray` the builder immediately unwraps, and once for
/// `append_varbinview_array` to rewrite every view so its buffer index is rebased onto the
/// builder's. Numbering the buffer up front instead makes the whole append one view per row plus
/// pushing the byte buffer.
fn append_to_varbinview(
    array: ArrayView<'_, VarBin>,
    builder: &mut VarBinViewBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let len = array.as_ref().len();
    let validity = array.varbin_validity().execute_mask(len, ctx)?;

    // Build the views against the index the pushed buffer will land at, so the builder does not
    // have to rebase them afterwards.
    let next_buffer_index = builder.completed_block_count() + u32::from(builder.in_progress());

    let parts = array.into_owned().into_data_parts();
    let offsets = parts.offsets.execute::<PrimitiveArray>(ctx)?;
    let (buffers, views) = varbin_decode_views(&offsets, parts.bytes, next_buffer_index);

    builder.push_buffer_and_adjusted_views(&buffers, &views, validity);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct VarBin;
