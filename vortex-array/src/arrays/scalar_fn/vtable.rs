// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::metadata::ScalarFnMetadata;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::vtable::{ArrayId, ArrayVTable, VTable};
use crate::{functions, vtable};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Vector;

vtable!(ScalarFn);

#[derive(Debug)]
pub struct ScalarFnVTable {
    vtable: functions::ScalarFnVTable,
}

impl VTable for ScalarFnVTable {
    type Array = ScalarFnArray;
    type Metadata = ScalarFnMetadata;
    type ArrayVTable = Self;
    type CanonicalVTable = ();
    type OperationsVTable = ();
    type ValidityVTable = ();
    type VisitorVTable = ();
    type ComputeVTable = ();
    type EncodeVTable = ();
    type OperatorVTable = ();

    fn id(&self) -> ArrayId {
        self.vtable.id()
    }

    fn encoding(array: &Self::Array) -> ArrayVTable {
        todo!()
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        todo!()
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        todo!()
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata> {
        todo!()
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &ScalarFnMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        let children: Vec<_> = metadata
            .child_dtypes
            .iter()
            .enumerate()
            .map(|(idx, child_dtype)| children.get(idx, child_dtype, len))
            .try_collect()?;

        todo!()
    }

    fn execute(array: &Self::Array, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        todo!()
    }
}
