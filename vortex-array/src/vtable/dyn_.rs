// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::mem::transmute;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::serde::ArrayChildren;
use crate::vtable::EncodeVTable;
use crate::vtable::VTable;

/// ArrayId is a globally unique name for the array's vtable.
pub type ArrayId = ArcRef<str>;
pub type ArrayVTable = ArcRef<dyn DynVTable>;

/// Dynamically typed trait for invoking array vtables.
pub trait DynVTable: 'static + private::Sealed + Send + Sync + Debug {
    /// Downcast the encoding to [`Any`].
    fn as_any(&self) -> &dyn Any;

    /// Returns the ID of the encoding.
    fn id(&self) -> ArrayId;

    /// Build an array from its parts.
    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArrayRef>;

    fn with_children(
        &self,
        array: &dyn Array,
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArrayRef>;

    /// Encode the canonical array into this encoding implementation.
    /// Returns `None` if this encoding does not support the given canonical array, for example
    /// if the data type is incompatible.
    ///
    /// Panics if `like` is encoded with a different encoding.
    fn encode(&self, input: &Canonical, like: Option<&dyn Array>)
    -> VortexResult<Option<ArrayRef>>;
}

/// Adapter struct used to lift the [`VTable`] trait into an object-safe [`DynVTable`]
/// implementation.
///
/// Since this is a unit struct with `repr(transparent)`, we are able to turn un-adapted array
/// structs into [`DynVTable`] using some cheeky casting inside [`std::ops::Deref`] and
/// [`AsRef`]. See the `vtable!` macro for more details.
#[repr(transparent)]
pub struct ArrayVTableAdapter<V: VTable>(V);

impl<V: VTable> DynVTable for ArrayVTableAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ArrayId {
        V::id(&self.0)
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata_bytes: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArrayRef> {
        let metadata = V::deserialize(metadata_bytes)?;
        let array = V::build(&self.0, dtype, len, &metadata, buffers, children)?;
        assert_eq!(array.len(), len, "Array length mismatch after building");
        assert_eq!(array.dtype(), dtype, "Array dtype mismatch after building");
        Ok(array.to_array())
    }

    fn with_children(
        &self,
        array: &dyn Array,
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArrayRef> {
        let buffers: Vec<BufferHandle> = array
            .buffers()
            .into_iter()
            .map(BufferHandle::Buffer)
            .collect();
        V::build(
            &self.0,
            array.dtype(),
            array.len(),
            &V::metadata(array.as_::<V>())?,
            &buffers,
            children,
        )
        .map(|a| a.into_array())
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
                        "Like array {} does not match requested encoding {}",
                        like.encoding_id(),
                        self.id()
                    )
                })
            })
            .transpose()?;

        let Some(array) =
            <V::EncodeVTable as EncodeVTable<V>>::encode(&self.0, input, downcast_like)?
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
}

impl<V: VTable> Debug for ArrayVTableAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoding").field("id", &self.id()).finish()
    }
}

impl Display for dyn DynVTable + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl PartialEq for dyn DynVTable + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn DynVTable + '_ {}

impl dyn DynVTable + '_ {
    pub fn as_<V: VTable>(&self) -> &V {
        self.as_any()
            .downcast_ref::<ArrayVTableAdapter<V>>()
            .map(|e| &e.0)
            .vortex_expect("Encoding is not of the expected type")
    }
}

pub trait ArrayVTableExt {
    /// Wraps the vtable into an `ArrayVTable` by static reference.
    fn as_vtable(&'static self) -> ArrayVTable;

    /// Wraps the vtable into an `ArrayVTable` by owned reference.
    fn into_vtable(self) -> ArrayVTable;
}

impl<V: VTable> ArrayVTableExt for V {
    fn as_vtable(&'static self) -> ArrayVTable {
        let dyn_vtable: &'static ArrayVTableAdapter<V> =
            unsafe { transmute::<&'static V, &'static ArrayVTableAdapter<V>>(self) };
        ArrayVTable::new_ref(dyn_vtable)
    }

    fn into_vtable(self) -> ArrayVTable {
        ArrayVTable::new_arc(Arc::new(ArrayVTableAdapter(self)))
    }
}

mod private {
    use crate::vtable::ArrayVTableAdapter;
    use crate::vtable::VTable;

    pub trait Sealed {}
    impl<V: VTable> Sealed for ArrayVTableAdapter<V> {}
}
