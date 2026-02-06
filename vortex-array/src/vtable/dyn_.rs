// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayAdapter;
use crate::ArrayRef;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::vtable::VTable;

/// ArrayId is a globally unique name for the array's vtable.
pub type ArrayId = ArcRef<str>;

/// Dynamically typed vtable trait.
///
/// This trait is sealed, therefore users should implement the strongly typed [`VTable`] trait
/// instead. The [`ArrayVTableExt::vtable`] function can be used to lift the implementation into
/// this object-safe form.
///
/// This trait contains the implementation API for Vortex arrays, allowing us to keep the public
/// [`Array`] trait API to a minimum.
pub trait DynVTable: 'static + private::Sealed + Send + Sync + Debug {
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        id: ArrayId,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef>;
    fn with_children(&self, array: &dyn Array, children: Vec<ArrayRef>) -> VortexResult<ArrayRef>;

    /// See [`VTable::reduce`]
    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>>;

    /// See [`VTable::reduce_parent`]
    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;

    /// See [`VTable::execute`]
    fn execute(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;

    /// See [`VTable::execute_parent`]
    fn execute_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapter struct used to lift the [`VTable`] trait into an object-safe [`DynVTable`]
/// implementation.
struct ArrayVTableAdapter<V: VTable>(PhantomData<V>);

impl<V: VTable> DynVTable for ArrayVTableAdapter<V> {
    fn build(
        &self,
        _id: ArrayId,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let metadata = V::deserialize(metadata, dtype, len, session)?;
        let array = V::build(dtype, len, &metadata, buffers, children)?;
        assert_eq!(array.len(), len, "Array length mismatch after building");
        assert_eq!(array.dtype(), dtype, "Array dtype mismatch after building");
        Ok(array.to_array())
    }

    fn with_children(&self, array: &dyn Array, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let mut array = array.as_::<V>().clone();
        V::with_children(&mut array, children)?;
        Ok(array.to_array())
    }

    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(reduced) = V::reduce(downcast::<V>(array))? else {
            return Ok(None);
        };
        vortex_ensure!(
            reduced.len() == array.len(),
            "Reduced array length mismatch from {} to {}",
            array.encoding_id(),
            reduced.encoding_id()
        );
        vortex_ensure!(
            reduced.dtype() == array.dtype(),
            "Reduced array dtype mismatch from {} to {}",
            array.encoding_id(),
            reduced.encoding_id()
        );
        Ok(Some(reduced))
    }

    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(reduced) = V::reduce_parent(downcast::<V>(array), parent, child_idx)? else {
            return Ok(None);
        };

        vortex_ensure!(
            reduced.len() == parent.len(),
            "Reduced array length mismatch from {} to {}",
            parent.encoding_id(),
            reduced.encoding_id()
        );
        vortex_ensure!(
            reduced.dtype() == parent.dtype(),
            "Reduced array dtype mismatch from {} to {}",
            parent.encoding_id(),
            reduced.encoding_id()
        );

        Ok(Some(reduced))
    }

    fn execute(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let result = V::execute(downcast::<V>(array), ctx)?;

        if cfg!(debug_assertions) {
            vortex_ensure!(
                result.as_ref().len() == array.len(),
                "Result length mismatch for {:?}",
                self
            );
            vortex_ensure!(
                result.as_ref().dtype() == array.dtype(),
                "Executed canonical dtype mismatch for {:?}",
                self
            );
        }

        // TODO(ngates): do we want to do this on every execution? We used to in to_canonical.
        result
            .as_ref()
            .statistics()
            .inherit_from(array.statistics());

        Ok(result)
    }

    fn execute_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(result) = V::execute_parent(downcast::<V>(array), parent, child_idx, ctx)? else {
            return Ok(None);
        };

        if cfg!(debug_assertions) {
            vortex_ensure!(
                result.as_ref().len() == parent.len(),
                "Executed parent canonical length mismatch"
            );
            vortex_ensure!(
                result.as_ref().dtype() == parent.dtype(),
                "Executed parent canonical dtype mismatch"
            );
        }

        Ok(Some(result))
    }
}

fn downcast<V: VTable>(array: &ArrayRef) -> &V::Array {
    array
        .as_any()
        .downcast_ref::<ArrayAdapter<V>>()
        .vortex_expect("Failed to downcast array to expected encoding type")
        .as_inner()
}

impl<V: VTable> Debug for ArrayVTableAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Encoding<{}>", type_name::<V>())
    }
}

impl<V: VTable> From<V> for &'static dyn DynVTable {
    fn from(_vtable: V) -> Self {
        const { &ArrayVTableAdapter::<V>(PhantomData) }
    }
}

pub trait ArrayVTableExt {
    /// Wraps the vtable into an [`DynVTable`] by static reference.
    fn vtable() -> &'static dyn DynVTable;
}

impl<V: VTable> ArrayVTableExt for V {
    fn vtable() -> &'static dyn DynVTable {
        const { &ArrayVTableAdapter::<V>(PhantomData) }
    }
}

mod private {
    use super::ArrayVTableAdapter;
    use crate::vtable::VTable;

    pub trait Sealed {}
    impl<V: VTable> Sealed for ArrayVTableAdapter<V> {}
}
