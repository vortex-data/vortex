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
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::TypedScalarFnInstance;
use crate::serde::ArrayChildren;

/// An adapter for enabling a scalar function to be serialized as an array.
pub struct ScalarFnArrayPlugin<V: ScalarFnVTable>(V);

impl<V: ScalarFnVTable> ScalarFnArrayPlugin<V> {
    /// Create a new plugin for the given scalar function vtable.
    pub fn new(vtable: V) -> Self {
        Self(vtable)
    }
}

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

    /// Deserialize a scalar function array from its serialized components.
    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>>;
}

/// The parts used to construct a ScalarFnArray.
pub struct ScalarFnArrayParts<V: ScalarFnVTable> {
    pub options: V::Options,
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
        let parts = <V as ScalarFnArrayVTable>::deserialize(
            &self.0, dtype, len, metadata, children, session,
        )?;
        Ok(ScalarFnArray::try_new_with_len(
            TypedScalarFnInstance::new(self.0.clone(), parts.options).erased(),
            parts.children,
            len,
        )?
        .into_array())
    }
}
