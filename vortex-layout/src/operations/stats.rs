use vortex_array::ArrayData;

use crate::operations::Operator;

pub struct ScanOperator;

impl Operator for ScanOperator {
    type Result = ArrayData;
}
