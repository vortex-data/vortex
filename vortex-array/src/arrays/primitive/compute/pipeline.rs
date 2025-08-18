use std::sync::Arc;

use vortex_error::{VortexResult, vortex_bail};
use vortex_vector::operators::Operator;
use vortex_vector::operators::primitive::PrimitiveOperator;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::pipeline::PipelineVTable;
use crate::vtable::ValidityHelper;

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Option<Arc<dyn Operator>>> {
        if !array.validity().all_valid()? {
            vortex_bail!(
                "PipelineVTable::to_operator is not supported for arrays with invalid values"
            );
        }
        Ok(Some(Arc::new(PrimitiveOperator::new(
            array.ptype(),
            array.byte_buffer().clone(),
        ))))
    }
}
