// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;
use std::sync::Arc;

use kernel::PARENT_KERNELS;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewData;
use crate::arrays::varbinview::array::NUM_SLOTS;
use crate::arrays::varbinview::array::SLOT_NAMES;
use crate::arrays::varbinview::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
mod kernel;
mod operations;
mod validity;
vtable!(VarBinView, VarBinView, VarBinViewData);

#[derive(Clone, Debug)]
pub struct VarBinView;

impl VarBinView {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.varbinview");
}

impl VTable for VarBinView {
    type ArrayData = VarBinViewData;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &VarBinViewData) -> &Self {
        &VarBinView
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &VarBinViewData) -> usize {
        array.views_handle().len() / size_of::<BinaryView>()
    }

    fn dtype(array: &VarBinViewData) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinViewData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &VarBinViewData,
        state: &mut H,
        precision: Precision,
    ) {
        for buffer in array.buffers.iter() {
            buffer.array_hash(state, precision);
        }
        array.views.array_hash(state, precision);
        array.validity().array_hash(state, precision);
    }

    fn array_eq(array: &VarBinViewData, other: &VarBinViewData, precision: Precision) -> bool {
        array.buffers.len() == other.buffers.len()
            && array
                .buffers
                .iter()
                .zip(other.buffers.iter())
                .all(|(a, b)| a.array_eq(b, precision))
            && array.views.array_eq(&other.views, precision)
            && array.validity().array_eq(&other.validity(), precision)
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.data_buffers().len() + 1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        let ndata = array.data_buffers().len();
        if idx < ndata {
            array.data_buffers()[idx].clone()
        } else if idx == ndata {
            array.views_handle().clone()
        } else {
            vortex_panic!("VarBinViewArray buffer index {idx} out of bounds")
        }
    }

    fn buffer_name(array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        let ndata = array.data_buffers().len();
        if idx < ndata {
            Some(format!("buffer_{idx}"))
        } else if idx == ndata {
            Some("views".to_string())
        } else {
            vortex_panic!("VarBinViewArray buffer_name index {idx} out of bounds")
        }
    }

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<VarBinViewData> {
        let Some((views_handle, data_handles)) = buffers.split_last() else {
            vortex_bail!("Expected at least 1 buffer, got 0");
        };

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 children, got {}", children.len());
        };

        let views_nbytes = views_handle.len();
        let expected_views_nbytes = len
            .checked_mul(size_of::<BinaryView>())
            .ok_or_else(|| vortex_err!("views byte length overflow for len={len}"))?;
        if views_nbytes != expected_views_nbytes {
            vortex_bail!(
                "Expected views buffer length {} bytes, got {} bytes",
                expected_views_nbytes,
                views_nbytes
            );
        }

        // If any buffer is on device, skip host validation and use try_new_handle.
        if buffers.iter().any(|b| b.is_on_device()) {
            return VarBinViewData::try_new_handle(
                views_handle.clone(),
                Arc::from(data_handles.to_vec()),
                dtype.clone(),
                validity,
            );
        }

        let data_buffers = data_handles
            .iter()
            .map(|b| b.as_host().clone())
            .collect::<Vec<_>>();
        let views = Buffer::<BinaryView>::from_byte_buffer(views_handle.clone().as_host().clone());

        VarBinViewData::try_new(views, Arc::from(data_buffers), dtype.clone(), validity)
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "VarBinViewArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::assert_arrays_eq;
    use crate::serde::ArrayParts;
    use crate::serde::SerializeOptions;

    #[test]
    fn test_nullable_varbinview_serde_roundtrip() {
        let array = VarBinViewArray::from_iter_nullable_str([
            Some("hello"),
            None,
            Some("world"),
            None,
            Some("a moderately long string for testing"),
        ]);
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array
            .clone()
            .into_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = ArrayParts::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(
                &dtype,
                len,
                &ReadContext::new(ctx.to_ids()),
                &LEGACY_SESSION,
            )
            .unwrap();

        assert_arrays_eq!(decoded, array);
    }
}
