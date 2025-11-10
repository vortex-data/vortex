// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vxo::{Array2, ArrayView};
use crate::EncodingId;
use arcref::ArcRef;
use std::any::Any;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use vortex_error::{vortex_err, VortexResult};

/// Non-object-safe VTable for Vortex arrays.
pub trait VTable: 'static + Sized + Send + Sync {
    /// The type of any instance data for the array.
    type Instance: 'static + Send + Sync + Debug + PartialEq + Eq + Hash;

    /// Returns the encoding ID for this VTable.
    fn id(&self) -> EncodingId;

    /// Validate the metadata, children, and buffers for the array.
    fn validate(&self, expr: &ArrayView<Self>) -> VortexResult<()>;

    //
    // /// Serialize the metadata for the expression.
    // ///
    // /// Should return `Ok(None)` if the expression is not serializable, and `Ok(vec![])` if it is
    // /// serializable but has no metadata.
    // fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
    //     Ok(None)
    // }
    //
    // /// Deserialize an instance of this expression.
    // ///
    // /// Returns `Ok(None)` if the expression is not serializable.
    // fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
    //     Ok(None)
    // }

    // /// Returns the name of the nth child of the expr.
    // fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName;
    //
    // /// Format this expression in nice human-readable SQL-style format
    // ///
    // /// The implementation should recursively format child expressions by calling
    // /// `expr.child(i).fmt_sql(f)`.
    // fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> fmt::Result;
    //
    // /// Format only the instance data for this expression.
    // ///
    // /// Defaults to a debug representation of the instance data.
    // #[allow(clippy::use_debug)]
    // fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> fmt::Result {
    //     write!(f, "{:?}", instance)
    // }
}

pub trait DynVTable: 'static + Send + Sync + private::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> EncodingId;
    fn validate(&self, array: &Array2) -> VortexResult<()>;
}

#[repr(transparent)]
pub struct VTableAdapter<V>(V);

impl<V: VTable> DynVTable for VTableAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn id(&self) -> EncodingId {
        self.0.id()
    }

    fn validate(&self, array: &Array2) -> VortexResult<()> {
        let view = ArrayView::<V>::maybe_new(array)
            .ok_or_else(|| vortex_err!("Failed to downcast array for validation"))?;
        V::validate(self, &view)
    }
}

mod private {
    use super::{VTable, VTableAdapter};

    pub trait Sealed {}
    impl<V: VTable> Sealed for VTableAdapter<V> {}
}

/// A type-erased Vortex array vtable.
#[derive(Clone)]
pub struct ArrayVTable(ArcRef<dyn DynVTable>);

impl ArrayVTable {
    pub(super) fn as_dyn(&self) -> &dyn DynVTable {
        self.0.as_ref()
    }

    /// Creates a new [`ArrayVTable`] from a static reference to a vtable.
    pub const fn from_static<V: VTable>(vtable: &'static V) -> Self {
        // SAFETY: We can safely cast the vtable to a VTableAdapter since it has the same layout.
        let adapted: &'static VTableAdapter<V> =
            unsafe { &*(vtable as *const V as *const VTableAdapter<V>) };
        Self(ArcRef::new_ref(adapted as &'static dyn DynVTable))
    }

    /// Returns the ID of this vtable.
    pub fn id(&self) -> EncodingId {
        self.0.id()
    }

    /// Returns whether this vtable is of a given type.
    pub fn is<V: VTable>(&self) -> bool {
        self.0.as_any().is::<VTableAdapter<V>>()
    }

    /// Returns the typed VTable for this expression.
    pub fn as_opt<V: VTable>(&self) -> Option<&V> {
        self.0
            .as_any()
            .downcast_ref::<VTableAdapter<V>>()
            .map(|adapter| &adapter.0)
    }
}

impl PartialEq for ArrayVTable {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}
impl Eq for ArrayVTable {}

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
