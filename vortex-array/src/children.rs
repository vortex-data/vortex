use vortex_error::VortexResult;

use crate::visitor::ArrayVisitor;
use crate::Array;

#[derive(Default, Debug)]
pub struct ChildrenCollector {
    children: Vec<Array>,
}

#[derive(Default, Debug)]
pub struct NamedChildrenCollector {
    children: Vec<(String, Array)>,
}

impl ChildrenCollector {
    pub fn children(self) -> Vec<Array> {
        self.children
    }
}

impl NamedChildrenCollector {
    pub fn children(self) -> Vec<(String, Array)> {
        self.children
    }
}

impl ArrayVisitor for ChildrenCollector {
    fn visit_child(&mut self, _name: &str, array: &Array) -> VortexResult<()> {
        self.children.push(array.clone());
        Ok(())
    }
}

impl ArrayVisitor for NamedChildrenCollector {
    fn visit_child(&mut self, name: &str, array: &Array) -> VortexResult<()> {
        self.children.push((name.to_string(), array.clone()));
        Ok(())
    }
}
