// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::layout::Layout;
use crate::v2::view::LayoutView;
use crate::LayoutId;
use arcref::ArcRef;
use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

pub type ChildName = ArcRef<str>;

pub trait VTable: 'static + Sized + Send + Sync {
    /// Instance data for this layout.
    type Instance: 'static + Send + Sync + Debug + PartialEq + Eq + Hash;

    /// Returns the ID of this layout.
    fn id(&self) -> LayoutId;

    /// Serializes the instance data into bytes.
    ///
    /// Returns `Ok(None)` if serialization is not supported, and `Ok(Some(vec![]))` if the layout
    /// is serializable but has no metadata.
    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Deserializes the instance data from bytes.
    fn deserialize(&self, _data: &[u8]) -> VortexResult<Self::Instance> {
        vortex_bail!("Layout {} is not deserializable", self.id())
    }

    /// Returns the name of the nth layout child, if applicable.
    fn child_name(&self, _view: &LayoutView<Self>, _child_idx: usize) -> ChildName;
}

/// A type-erased vtable for dynamic layouts.
pub trait DynLayoutVTable: 'static + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn id(&self) -> LayoutId;
    fn serialize(&self, instance: &dyn Any) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize(&self, data: &[u8]) -> VortexResult<Box<dyn Any>>;
    fn child_name(&self, layout: &Layout, child_idx: usize) -> ChildName;
}

struct LayoutVTableAdapter<V: VTable>(V);
impl<V: VTable> DynLayoutVTable for LayoutVTableAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> LayoutId {
        V::id(&self.0)
    }

    fn serialize(&self, instance: &dyn Any) -> VortexResult<Option<Vec<u8>>> {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast layout instance to expected type");
        V::serialize(&self.0, instance)
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Box<dyn Any + Send>> {
        Ok(Box::new(V::deserialize(&self.0, metadata)?))
    }

    fn child_name(&self, instance: &dyn Any, child_idx: usize) -> vortex_array::expr::ChildName {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast layout instance to expected type");
        V::child_name(&self.0, instance, child_idx)
    }
}
