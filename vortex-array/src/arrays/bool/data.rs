use vortex_error::VortexError;

use crate::array::data::ArrayData;
use crate::arrays::BoolArray;

pub struct BoolMetadata {}

impl TryFrom<ArrayData> for BoolArray {
    type Error = VortexError;

    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl TryFrom<&BoolArray> for ArrayData {
    type Error = VortexError;

    fn try_from(value: &BoolArray) -> Result<Self, Self::Error> {
        todo!()
    }
}
