// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::varbinview::VarBinViewData;
use crate::arrays::varbinview::array::VarBinViewSlots;
use crate::arrays::varbinview::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::match_each_varbin_builder;
mod kernel;
mod operations;
mod plugin;
mod validity;
/// A [`VarBinView`]-encoded Vortex array.
pub type VarBinViewArray = Array<VarBinView>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

#[derive(Clone, Debug)]
pub struct VarBinView;

impl ArrayHash for VarBinViewData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        for buffer in self.buffers.iter() {
            buffer.array_hash(state, accuracy);
        }
        self.views.array_hash(state, accuracy);
    }
}

impl ArrayEq for VarBinViewData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.buffers.len() == other.buffers.len()
            && self
                .buffers
                .iter()
                .zip(other.buffers.iter())
                .all(|(a, b)| a.array_eq(b, accuracy))
            && self.views.array_eq(&other.views, accuracy)
    }
}

impl VTable for VarBinView {
    type TypedArrayData = VarBinViewData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.varbinview");
        *ID
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.data_buffers().len() + 1
    }

    fn validate(
        &self,
        data: &VarBinViewData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == VarBinViewSlots::COUNT,
            "VarBinViewArray expected {} slots, found {}",
            VarBinViewSlots::COUNT,
            slots.len()
        );
        vortex_ensure!(
            data.len() == len,
            "VarBinViewArray length {} does not match outer length {}",
            data.len(),
            len
        );
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            "VarBinViewArray dtype must be binary or utf8, got {dtype}"
        );
        Ok(())
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let Some((views, data_buffers)) = buffers.split_last() else {
            vortex_bail!("Expected at least 1 buffer, got 0");
        };
        let data = VarBinViewData::try_new_handle(
            views.clone(),
            Arc::from(data_buffers.to_vec()),
            array.dtype().clone(),
            array.validity()?,
        )?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        VarBinViewSlots::NAMES[idx].to_string()
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
        if let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() {
            return builder.append_varbinview_array(&array.into_owned(), ctx);
        }
        if let Some(result) =
            match_each_varbin_builder!(builder, |builder| builder.append_varbinview(array, ctx))
        {
            return result;
        }
        vortex_bail!("append_to_builder for VarBinView requires a variable-binary builder")
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
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::assert_arrays_eq;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;

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

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let array_ctx = ArrayContext::empty();
        let serialized = array
            .clone()
            .into_array()
            .serialize(&array_ctx, &session, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)
            .unwrap();

        assert_arrays_eq!(decoded, array, &mut ctx);
    }
}
