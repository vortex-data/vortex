use vortex::dtype::DType;
use vortex::error::VortexError;

use crate::duckdb::LogicalType;

impl TryFrom<&DType> for LogicalType {
    type Error = VortexError;

    fn try_from(_value: &DType) -> Result<Self, Self::Error> {
        todo!()
    }
}
