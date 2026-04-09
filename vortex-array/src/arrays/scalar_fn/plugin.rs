// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFn;
use crate::scalar_fn::ScalarFnVTable;
use crate::serde::ArrayChildren;

/// An adapter for enabling a scalar function to be serialized as an array.
pub struct ScalarFnArrayPlugin<V: ScalarFnVTable>(V);

pub trait ScalarFnArrayVTable: ScalarFnVTable {
    /// Serialize metadata for storing the scalar function as an array.
    ///
    /// Notably, this metadata needs enough information to reconstruct the child DTypes, as well
    /// as the scalar function's own options.
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>>;

    /// Deserialize a scalar function array from the
    fn deserialize(
        &self,
        dtype: &DType,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>>;
}

/// The parts used to construct a ScalarFnArray.
pub struct ScalarFnArrayParts<V: ScalarFnVTable> {
    pub scalar_fn: ScalarFn<V>,
    pub children: Vec<ArrayRef>,
}

impl<V: ScalarFnVTable + ScalarFnArrayVTable> ArrayPlugin for ScalarFnArrayPlugin<V> {
    fn id(&self) -> ArrayId {
        self.0.id()
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // We serialize the scalar function options, along with any scalar function array data.
        let scalar_fn = array.as_::<ExactScalarFn<V>>();
        <V as ScalarFnArrayVTable>::serialize(&self.0, &scalar_fn, session)
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let parts =
            <V as ScalarFnArrayVTable>::deserialize(&self.0, dtype, metadata, children, session)?;
        Ok(ScalarFnArray::try_new(parts.scalar_fn.erased(), parts.children, len)?.into_array())
    }
}
