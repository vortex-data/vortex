use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::traversal::{MutNodeVisitor, TransformResult};
use crate::{ExprRef, Select};

/// Select is a useful expression, however it can be defined in terms of get_item & pack,
/// once the expression type is known, this simplifications pass removes the select expression.
pub struct RemoveSelectTransform{
    ident_dtype: DType
}

impl RemoveSelectTransform {
    pub fn new(ident_dtype: DType) -> Self {
        Self { ident_dtype }
    }


impl MutNodeVisitor for RemoveSelectTransform {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: ExprRef) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(node) = node.as_any().downcast_ref::<Select>() {
            let ident = node.ident();
            let index = node.index();
            let ident_dtype = ident.dtype();
            if ident_dtype.eq_ignore_nullability(&self.ident_dtype) {
                let new_node = get_item(ident.clone(), index.clone());
                Ok(TransformResult::Replace(new_node))
            } else {
                Ok(TransformResult::Continue)
            }
        } else {
            Ok(TransformResult::Continue)
        }
    }
}
