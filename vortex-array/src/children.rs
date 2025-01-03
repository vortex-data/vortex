use vortex_error::VortexResult;

use crate::visitor::ArrayVisitor;
use crate::ArrayData;

#[derive(Default, Debug)]
pub struct ChildrenCollector {
    children: Vec<ArrayData>,
}

#[derive(Default, Debug)]
pub struct NamedChildrenCollector {
    children: Vec<(String, ArrayData)>,
}

impl ChildrenCollector {
    pub fn children(self) -> Vec<ArrayData> {
        self.children
    }
}

impl NamedChildrenCollector {
    pub fn children(self) -> Vec<(String, ArrayData)> {
        self.children
    }
}

impl ArrayVisitor for ChildrenCollector {
    fn visit_child(&mut self, _name: &str, array: &ArrayData) -> VortexResult<()> {
        self.children.push(array.clone());
        Ok(())
    }
}

impl ArrayVisitor for NamedChildrenCollector {
    fn visit_child(&mut self, name: &str, array: &ArrayData) -> VortexResult<()> {
        self.children.push((name.to_string(), array.clone()));
        Ok(())
    }
}
