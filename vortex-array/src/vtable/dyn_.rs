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
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::Array;
use crate::ArrayAdapter;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::vtable::EncodeVTable;
use crate::vtable::VTable;

/// ArrayId is a globally unique name for the array's vtable.
pub type ArrayId = ArcRef<str>;

/// Dynamically typed vtable trait.
///
/// This trait is sealed, therefore users should implement the strongly typed [`VTable`] trait
/// instead. The [`ArrayVTableExt::as_vtable`] can be used to lift the implementation into this
/// object-safe form.
///
/// This trait contains the implementation API for Vortex arrays, allowing us to keep the public
/// [`Array`] trait API to a minimum.
pub trait DynVTable: 'static + private::Sealed + Send + Sync + Debug {
    fn build(
        &self,
        id: ArrayId,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArrayRef>;
    /// See [`super::EncodeVTable::encode`]
    fn encode(&self, input: &Canonical, like: Option<&dyn Array>)
    -> VortexResult<Option<ArrayRef>>;

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
    fn execute_canonical(
        &self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Canonical>;

    /// See [`VTable::execute_parent`]
    fn execute_canonical_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>>;
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
    ) -> VortexResult<ArrayRef> {
        let metadata = V::deserialize(metadata)?;
        let array = V::build(dtype, len, &metadata, buffers, children)?;
        assert_eq!(array.len(), len, "Array length mismatch after building");
        assert_eq!(array.dtype(), dtype, "Array dtype mismatch after building");
        Ok(array.to_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let downcast_like = like
            .map(|like| {
                like.as_opt::<V>().ok_or_else(|| {
                    vortex_err!(
                        "Like array {} does not match requested encoding {:?}",
                        like.encoding_id(),
                        self
                    )
                })
            })
            .transpose()?;

        let Some(array) = <V::EncodeVTable as EncodeVTable<V>>::encode(input, downcast_like)?
        else {
            return Ok(None);
        };

        let input = input.as_ref();
        if array.len() != input.len() {
            vortex_bail!(
                "Array length mismatch after encoding: {} != {}",
                array.len(),
                input.len()
            );
        }
        if array.dtype() != input.dtype() {
            vortex_bail!(
                "Array dtype mismatch after encoding: {} != {}",
                array.dtype(),
                input.dtype()
            );
        }

        Ok(Some(array.into_array()))
    }

    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(reduced) = V::reduce(downcast::<V>(array))? else {
            return Ok(None);
        };
        vortex_ensure!(
            reduced.len() == array.len(),
            "Reduced array length mismatch from {} to {}",
            array.display_tree(),
            reduced.display_tree()
        );
        vortex_ensure!(
            reduced.dtype() == array.dtype(),
            "Reduced array dtype mismatch from {} to {}",
            array.display_tree(),
            reduced.display_tree()
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
            parent.display_tree(),
            reduced.display_tree()
        );
        vortex_ensure!(
            reduced.dtype() == parent.dtype(),
            "Reduced array dtype mismatch from {} to {}",
            parent.display_tree(),
            reduced.display_tree()
        );

        Ok(Some(reduced))
    }

    fn execute_canonical(
        &self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Canonical> {
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

        Ok(result)
    }

    fn execute_canonical_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
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
        .vortex_expect("Invalid options type for expression")
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
