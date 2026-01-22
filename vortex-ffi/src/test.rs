// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::vtable::ArrayId;
use vortex::array::vtable::VTable;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::session::{SessionExt, VortexSession};

#[repr(C)]
struct FFIVTableOpaque {
    metadata: (array: * const u8) -> void_c,
}

struct FFIVTable {
    ffi_vtable: &'static FFIVTableOpaque,
}

impl VTable for FFIVTable {
    type Array = *const u8;
    type Metadata = ();
    type ArrayVTable = ();
    type OperationsVTable = ();
    type ValidityVTable = ();
    type VisitorVTable = ();
    type ComputeVTable = ();

    fn id(array: &Self::Array) -> ArrayId {
        todo!()
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        todo!()
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        todo!()
    }

    fn deserialize(id: ArrayId, bytes: &[u8], session: VortexSession) -> VortexResult<Self::Metadata> {
        let ffi_vtable: FFIVTableOpaque = session.get::<FFISession>().get_vtable(id);
        ffi_vtable.deserialzie()
        todo!()
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        todo!()
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        todo!()
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        todo!()
    }
}
