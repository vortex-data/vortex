use vortex_error::VortexResult;

use crate::visitor::ArrayVisitor;
use crate::{Array, ArrayRef};

#[derive(Default, Debug)]
pub struct ChildrenCollector {
    children: Vec<ArrayRef>,
}

#[derive(Default, Debug)]
pub struct NamedChildrenCollector {
    children: Vec<(String, ArrayRef)>,
}

impl ChildrenCollector {
    pub fn children(self) -> Vec<ArrayRef> {
        self.children
    }
}

impl NamedChildrenCollector {
    pub fn children(self) -> Vec<(String, ArrayRef)> {
        self.children
    }
}

impl ArrayVisitor for ChildrenCollector {
    fn visit_child(&mut self, _name: &str, array: &dyn Array) -> VortexResult<()> {
        self.children.push(array.to_array());
        Ok(())
    }
}

impl ArrayVisitor for NamedChildrenCollector {
    fn visit_child(&mut self, name: &str, array: &dyn Array) -> VortexResult<()> {
        self.children.push((name.to_string(), array.to_array()));
        Ok(())
    }
}
