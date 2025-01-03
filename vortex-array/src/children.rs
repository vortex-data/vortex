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

#[derive(Default, Debug)]
pub struct NamedTreeCollector {
    all_children: Vec<(String, ArrayData)>,
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

impl NamedTreeCollector {
    pub fn all_children(self) -> Vec<(String, ArrayData)> {
        self.all_children
    }

    pub fn visit_all_children(array: &ArrayData) -> VortexResult<Vec<(String, ArrayData)>> {
        let mut collector = NamedTreeCollector::default();
        array.encoding().accept(array, &mut collector)?;
        Ok(collector.all_children)
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

impl ArrayVisitor for NamedTreeCollector {
    fn visit_child(&mut self, name: &str, array: &ArrayData) -> VortexResult<()> {
        self.all_children.push((name.to_string(), array.clone()));
        array.encoding().accept(array, self)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::array::{ChunkedArray, ListArray, PrimitiveArray};
    use crate::validity::Validity;
    use crate::{IntoArrayData, NamedTreeCollector};

    #[test]
    fn nested_collect() {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array(),
            PrimitiveArray::from_iter(vec![0i32, 2, 3]).into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let chunk = ChunkedArray::from_iter(vec![list]).into_array();

        let children = NamedTreeCollector::visit_all_children(&chunk).unwrap();
        assert_eq!(children.len(), 4)
    }
}
