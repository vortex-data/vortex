use vortex_array::array::ConstantArray;
use vortex_array::compute::{like, LikeFn, LikeOptions};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl LikeFn<DictArray> for DictEncoding {
    fn like(
        &self,
        array: DictArray,
        pattern: &ArrayData,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayData>> {
        if let Some(pattern) = pattern.as_constant() {
            let pattern = ConstantArray::new(pattern, array.values().len()).into_array();
            let values = like(array.values(), &pattern, options)?;
            Ok(Some(
                DictArray::try_new(array.codes(), values)?.into_array(),
            ))
        } else {
            Ok(None)
        }
    }
}
