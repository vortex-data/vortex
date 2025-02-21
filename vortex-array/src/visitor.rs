//! Utilities to traverse array trees using the visitor pattern.

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::patches::Patches;
use crate::validity::Validity;
use crate::Array;

pub trait ArrayVisitor {
    /// Visit a child of this array.
    fn visit_child(&mut self, _name: &str, _array: &Array) -> VortexResult<()> {
        Ok(())
    }

    /// Utility for visiting Array validity.
    fn visit_validity(&mut self, validity: &Validity) -> VortexResult<()> {
        if let Some(v) = validity.as_array() {
            self.visit_child("validity", v)
        } else {
            Ok(())
        }
    }

    /// Utility for visiting Array patches.
    fn visit_patches(&mut self, patches: &Patches) -> VortexResult<()> {
        self.visit_child("patch_indices", patches.indices())?;
        self.visit_child("patch_values", patches.values())
    }

    fn visit_buffer(&mut self, _buffer: &ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}

/// Visitor to flatten an array tree.
#[derive(Default, Debug)]
pub struct ChildrenVisitor {
    pub children: Vec<Array>,
}

/// Visitor to flatten an array tree while keeping each child's name.
#[derive(Default, Debug)]
pub struct NamedChildrenVisitor {
    pub children: Vec<(String, Array)>,
}

impl ArrayVisitor for ChildrenVisitor {
    fn visit_child(&mut self, _name: &str, array: &Array) -> VortexResult<()> {
        self.children.push(array.clone());
        Ok(())
    }
}

impl ArrayVisitor for NamedChildrenVisitor {
    fn visit_child(&mut self, name: &str, array: &Array) -> VortexResult<()> {
        self.children.push((name.to_string(), array.clone()));
        Ok(())
    }
}
