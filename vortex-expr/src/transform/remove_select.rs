use itertools::Itertools;
use vortex_dtype::{DType, Field};
use vortex_error::{VortexExpect, VortexResult};

use crate::traversal::{MutNodeVisitor, TransformResult};
use crate::{get_item, pack, ExprRef, Select, SelectField, VortexExpr};

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
        if let Some(select) = node.as_any().downcast_ref::<Select>() {

            let select_dtype = select.return_dtype(&self.ident_dtype)?.as_struct().vortex_expect("select must return a struct");

            let new_fields = match select.fields() {
                SelectField::Include(fields) => fields,
                SelectField::Exclude(fields) => select_dtype.field_info(fields).map(|f_info| Field::Name(f_info.name))
            };

            let new_f = new_fields.into_iter().map(|f| f.clone().into_named_field(select_dtype.names())).collect::<VortexResult<Vec<_>>>()?;

            let new_expr = pack(new_f.iter().map(|f| f), select.expr().clone());
            let inner = new_f.into_iter().map(|f| get_item(f, ))
        }
        } else {
            Ok(TransformResult::no(node))
        }
    }
}
