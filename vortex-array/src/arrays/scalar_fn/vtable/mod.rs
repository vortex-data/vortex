// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod validity;
mod visitor;

use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::metadata::ScalarFnMetadata;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::vtable::{ArrayId, ArrayVTable, ArrayVTableExt, BaseArrayVTable, NotSupported, VTable};
use crate::{functions, vtable, Array};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_ensure, VortexResult};
use vortex_vector::Vector;

vtable!(ScalarFn);

#[derive(Clone, Debug)]
pub struct ScalarFnVTable {
    vtable: functions::ScalarFnVTable,
}

impl VTable for ScalarFnVTable {
    type Array = ScalarFnArray;
    type Metadata = ScalarFnMetadata;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = NotSupported;

    fn id(&self) -> ArrayId {
        self.vtable.id()
    }

    fn encoding(array: &Self::Array) -> ArrayVTable {
        array.vtable.clone()
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        let child_dtypes = array.children().iter().map(|c| c.dtype().clone()).collect();
        Ok(ScalarFnMetadata {
            scalar_fn: array.scalar_fn.clone(),
            child_dtypes,
        })
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // Not supported
        Ok(None)
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_bail!("Deserialization of ScalarFnVTable metadata is not supported");
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &ScalarFnMetadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        let children: Vec<_> = metadata
            .child_dtypes
            .iter()
            .enumerate()
            .map(|(idx, child_dtype)| children.get(idx, child_dtype, len))
            .try_collect()?;

        #[cfg(debug_assertions)]
        {
            let child_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
            vortex_ensure!(
                &metadata.scalar_fn.return_dtype(&child_dtypes)? == dtype,
                "Return dtype mismatch when building ScalarFnArray"
            );
        }

        Ok(ScalarFnArray {
            // This requires a new Arc, but we plan to remove this later anyway.
            vtable: self.to_vtable(),
            scalar_fn: metadata.scalar_fn.clone(),
            dtype: dtype.clone(),
            len,
            children,
            stats: Default::default(),
        })
    }

    fn execute(array: &Self::Array, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        let input_dtypes: Vec<_> = array.children().iter().map(|c| c.dtype().clone()).collect();
        let input_datums = array
            .children()
            .iter()
            .map(|child| child.execute())
            .try_collect()?;
        let ctx = functions::ExecutionCtx::new(
            array.len(),
            array.dtype.clone(),
            input_dtypes,
            input_datums,
        );
        array.scalar_fn.execute(&ctx)
    }
}
