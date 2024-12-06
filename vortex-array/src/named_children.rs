use vortex_error::VortexResult;

use crate::visitor::ArrayVisitor;
use crate::ArrayData;

#[derive(Default, Debug)]
pub struct NamedChildrenCollector {
    children: Vec<(String, ArrayData)>,
    depth: Option<usize>,
}

impl NamedChildrenCollector {
    pub fn new_with_depth(depth: usize) -> Self {
        Self {
            depth: Some(depth),
            ..Default::default()
        }
    }

    pub fn children(&self) -> &[(String, ArrayData)] {
        self.children.as_slice()
    }
}

impl ArrayVisitor for NamedChildrenCollector {
    fn visit_child(&mut self, name: &str, array: &ArrayData) -> VortexResult<()> {
        if let Some(depth) = self.depth.as_ref() {
            // Once the depth is 0, we stop collecting children.
            if *depth <= 0 {
                return Ok(());
            }
        }
        self.children.push((name.to_string(), array.clone()));
        self.depth = self.depth.map(|d| d - 1);
        Ok(())
    }
}
