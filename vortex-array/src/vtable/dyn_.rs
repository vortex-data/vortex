// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

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

/// Dynamically typed trait for invoking array vtables.
///
/// This trait contains the internal API for Vortex arrays, allowing us to expose things here
/// that we do not want to be part of the public [`Array`] trait.
pub trait DynVTable: 'static + private::Sealed + Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;

    fn id(&self) -> ArrayId;

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArrayRef>;
    fn with_children(&self, array: &dyn Array, children: Vec<ArrayRef>) -> VortexResult<ArrayRef>;
    fn encode(&self, input: &Canonical, like: Option<&dyn Array>)
    -> VortexResult<Option<ArrayRef>>;

    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;

    fn execute_canonical(
        &self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Canonical>;
    fn execute_canonical_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>>;

    fn slice(&self, array: &ArrayRef, range: Range<usize>) -> VortexResult<Option<ArrayRef>>;
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

    fn with_children(&self, array: &dyn Array, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let mut array = array.as_::<V>().clone();
        V::with_children(&mut array, children)?;
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
                "Result length mismatch for {}",
                self.id()
            );
            vortex_ensure!(
                result.as_ref().dtype() == array.dtype(),
                "Executed canonical dtype mismatch for {}",
                self.id()
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

    fn slice(&self, array: &ArrayRef, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        vortex_ensure!(
            range.end <= array.len(),
            "slice range {}..{} out of bounds for array of length {}",
            range.start,
            range.end,
            array.len()
        );

        let Some(sliced) = V::slice(downcast::<V>(array), range.clone())? else {
            return Ok(None);
        };
        vortex_ensure!(
            sliced.len() == range.len(),
            "Sliced array length mismatch: expected {}, got {}",
            range.len(),
            sliced.len()
        );
        vortex_ensure!(
            sliced.dtype() == array.dtype(),
            "Sliced array dtype mismatch: expected {}, got {}",
            array.dtype(),
            sliced.dtype()
        );
        Ok(Some(sliced))
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
        f.debug_struct("Encoding").field("id", &self.id()).finish()
    }
}

/// Dynamically typed array vtable.
#[derive(Clone)]
pub struct ArrayVTable(ArcRef<dyn DynVTable>);

impl ArrayVTable {
    /// Returns the underlying vtable API, public only within the crate.
    pub(crate) fn as_dyn(&self) -> &dyn DynVTable {
        self.0.as_ref()
    }

    /// Return the vtable as an Any reference.
    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    /// Creates a new [`ArrayVTable`] from a vtable.
    ///
    /// Prefer to use [`Self::new_static`] when possible.
    pub fn new<V: VTable>(vtable: V) -> Self {
        Self(ArcRef::new_arc(Arc::new(ArrayVTableAdapter(vtable))))
    }

    /// Creates a new [`ArrayVTable`] from a static reference to a vtable.
    pub const fn new_static<V: VTable>(vtable: &'static V) -> Self {
        // SAFETY: We can safely cast the vtable to a VTableAdapter since it has the same layout.
        let adapted: &'static ArrayVTableAdapter<V> =
            unsafe { &*(vtable as *const V as *const ArrayVTableAdapter<V>) };
        Self(ArcRef::new_ref(adapted as &'static dyn DynVTable))
    }

    /// Returns the ID of this vtable.
    pub fn id(&self) -> ArrayId {
        self.0.id()
    }

    /// Returns whether this vtable is of a given type.
    pub fn is<V: VTable>(&self) -> bool {
        self.0.as_any().is::<V>()
    }

    /// Encode the canonical array like the given array.
    pub fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        self.as_dyn().encode(input, like)
    }

    /// Slice the array using the VTable's slice implementation.
    pub fn slice(&self, array: &ArrayRef, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        self.as_dyn().slice(array, range)
    }
}

impl PartialEq for ArrayVTable {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}
impl Eq for ArrayVTable {}

impl Hash for ArrayVTable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
    }
}

impl Display for ArrayVTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_dyn().id())
    }
}

impl Debug for ArrayVTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_dyn().id())
    }
}

pub trait ArrayVTableExt {
    /// Wraps the vtable into an `ArrayVTable` by static reference.
    fn as_vtable(&'static self) -> ArrayVTable;

    /// Wraps the vtable into an `ArrayVTable` by owned reference.
    fn into_vtable(self) -> ArrayVTable;

    fn to_vtable(&self) -> ArrayVTable
    where
        Self: Clone;
}

// TODO(ngates): deprecate these functions in favor of `ArrayVTable::new` and
//  `ArrayVTable::new_static`.
impl<V: VTable> ArrayVTableExt for V {
    fn as_vtable(&'static self) -> ArrayVTable {
        ArrayVTable::new_static(self)
    }

    fn into_vtable(self) -> ArrayVTable {
        ArrayVTable::new(self)
    }

    fn to_vtable(&self) -> ArrayVTable
    where
        Self: Clone,
    {
        ArrayVTable::new(self.clone())
    }
}

mod private {
    use crate::vtable::ArrayVTableAdapter;
    use crate::vtable::VTable;

    pub trait Sealed {}
    impl<V: VTable> Sealed for ArrayVTableAdapter<V> {}
}
