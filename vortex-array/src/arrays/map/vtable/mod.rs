// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTableFromChild;
use crate::array::with_empty_buffers;
use crate::arrays::ListView;
use crate::arrays::map::MapData;
use crate::arrays::map::array::ENTRIES_SLOT;
use crate::arrays::map::array::NUM_SLOTS;
use crate::arrays::map::array::SLOT_NAMES;
use crate::arrays::map::array::validate_entries;
use crate::arrays::map::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::match_each_map_builder;
use crate::serde::ArrayChildren;

mod kernel;
mod operations;
mod validity;

/// A [`Map`]-encoded Vortex array.
pub type MapArray = Array<Map>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

/// The canonical encoding for [`DType::Map`].
///
/// A map array has one `ListView<Struct<key, value>>` child. Its outer dtype retains map-specific
/// metadata such as the `keys_sorted` assertion.
#[derive(Clone, Debug, Default)]
pub struct Map;

impl VTable for Map {
    type TypedArrayData = MapData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.map");
        *ID
    }

    fn validate(
        &self,
        _data: &MapData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "MapArray expected {NUM_SLOTS} slot, found {}",
            slots.len()
        );

        let DType::Map(map_dtype, nullability) = dtype else {
            vortex_bail!("Expected map dtype, got {dtype}");
        };
        let entries = slots[ENTRIES_SLOT]
            .as_ref()
            .ok_or_else(|| vortex_error::vortex_err!("MapArray missing entries slot"))?;
        validate_entries(map_dtype, *nullability, len, entries)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("MapArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
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
        if !metadata.is_empty() {
            vortex_bail!(
                "MapArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        vortex_ensure!(buffers.is_empty(), "MapArray expects no buffers");

        let DType::Map(map_dtype, nullability) = dtype else {
            vortex_bail!("Expected map dtype, got {dtype}");
        };
        vortex_ensure!(
            children.len() == NUM_SLOTS,
            "MapArray expected {NUM_SLOTS} child, found {}",
            children.len()
        );

        let expected_entries_dtype =
            DType::List(std::sync::Arc::new(map_dtype.entries_dtype()), *nullability);
        let entries = children.get(ENTRIES_SLOT, &expected_entries_dtype, len)?;
        vortex_ensure!(
            entries.is::<ListView>(),
            "MapArray entries must use vortex.listview encoding, got {}",
            entries.encoding_id()
        );

        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, MapData)
            .with_slots(smallvec![Some(entries)]))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match match_each_map_builder!(&mut *builder, |b| b.append_map_array(array, ctx)) {
            Some(result) => result,
            None => vortex_bail!(
                "cannot append a Map array of dtype {} to a {} builder",
                array.dtype(),
                builder.dtype()
            ),
        }
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
