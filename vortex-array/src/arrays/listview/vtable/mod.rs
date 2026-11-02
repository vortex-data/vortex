// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::with_empty_buffers;
use crate::arrays::listview::ListViewData;
use crate::arrays::listview::ListViewSlots;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_list_builder;
mod kernel;
mod operations;
mod plugin;
mod validity;
/// A [`ListView`]-encoded Vortex array.
pub type ListViewArray = Array<ListView>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

#[derive(Clone, Debug)]
pub struct ListView;

#[derive(Clone, prost::Message)]
pub struct ListViewMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(int32, tag = "2")]
    offset_ptype: i32,
    #[prost(int32, tag = "3")]
    size_ptype: i32,
}

impl ListViewMetadata {
    fn get_offset_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.offset_ptype).map_err(|_| {
            vortex_err!(
                "ListViewMetadata offset_ptype must be a known PType, got {}",
                self.offset_ptype
            )
        })
    }

    fn get_size_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.size_ptype).map_err(|_| {
            vortex_err!(
                "ListViewMetadata size_ptype must be a known PType, got {}",
                self.size_ptype
            )
        })
    }
}

impl ArrayHash for ListViewData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.is_zero_copy_to_list().hash(state);
    }
}

impl ArrayEq for ListViewData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.is_zero_copy_to_list() == other.is_zero_copy_to_list()
    }
}

impl VTable for ListView {
    type TypedArrayData = ListViewData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.listview");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ListViewArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ListViewArray buffer_name index {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn validate(
        &self,
        _data: &ListViewData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), ListViewSlots::COUNT);
        let elements = slots[ListViewSlots::ELEMENTS]
            .as_ref()
            .vortex_expect("ListViewArray elements slot");
        let offsets = slots[ListViewSlots::OFFSETS]
            .as_ref()
            .vortex_expect("ListViewArray offsets slot");
        let sizes = slots[ListViewSlots::SIZES]
            .as_ref()
            .vortex_expect("ListViewArray sizes slot");
        vortex_ensure!(
            offsets.len() == len && sizes.len() == len,
            "ListViewArray length {} does not match outer length {}",
            offsets.len(),
            len
        );

        let actual_dtype = DType::List(Arc::new(elements.dtype().clone()), dtype.nullability());
        vortex_ensure_eq!(
            &actual_dtype,
            dtype,
            "ListViewArray dtype does not match outer dtype",
        );

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ListViewSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    // The complexity comes from the expansion of `match_each_list_builder!`.
    #[expect(clippy::cognitive_complexity)]
    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match match_each_list_builder!(&mut *builder, |b| b.append_listview_array(array, ctx)) {
            Some(result) => result,
            None => vortex_bail!(
                "cannot append a ListView array of dtype {} to a {} builder",
                array.dtype(),
                builder.dtype()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use prost::Message;
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ListView;
    use super::ListViewMetadata;
    use crate::ArrayDeserialization;
    use crate::ArrayPlugin;
    use crate::ArrayRef;
    use crate::array::VTable;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::offset_ptype(
        |raw| ListViewMetadata { offset_ptype: raw, ..Default::default() },
        ListViewMetadata::get_offset_ptype,
        "ListViewMetadata offset_ptype must be a known PType, got 11",
    )]
    #[case::size_ptype(
        |raw| ListViewMetadata { size_ptype: raw, ..Default::default() },
        ListViewMetadata::get_size_ptype,
        "ListViewMetadata size_ptype must be a known PType, got 11",
    )]
    fn test_deserialize_rejects_unknown_ptype(
        #[case] with_ptype: fn(i32) -> ListViewMetadata,
        #[case] get_ptype: fn(&ListViewMetadata) -> VortexResult<PType>,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        let dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let metadata = with_ptype(11).encode_to_vec();
        let children: Vec<ArrayRef> = Vec::new();
        let result = ListView.deserialize(
            ArrayDeserialization::new(VTable::id(&ListView), &dtype, 0, &metadata, &[], &children),
            &SESSION,
        );
        assert!(result.is_err_and(|err| err.to_string().contains(expected)));

        assert_eq!(get_ptype(&with_ptype(PType::U32 as i32))?, PType::U32);

        Ok(())
    }
}
