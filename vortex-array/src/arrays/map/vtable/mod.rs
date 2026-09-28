// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::arrays::map::MapData;
use crate::arrays::map::MapSlots;
use crate::arrays::map::MapSlotsView;
use crate::arrays::map::array::validate_entries;
use crate::arrays::map::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::match_each_map_builder;

mod kernel;
mod operations;
mod plugin;
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
            slots.len() == MapSlots::COUNT,
            "MapArray expected {} slot, found {}",
            MapSlots::COUNT,
            slots.len()
        );

        let DType::Map(map_dtype, nullability) = dtype else {
            vortex_bail!("Expected map dtype, got {dtype}");
        };
        let slots = MapSlotsView::from_slots(slots);
        validate_entries(map_dtype, *nullability, len, slots.entries)
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        MapSlots::NAMES[idx].to_string()
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
