use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::ConstantArray;
use crate::compute::{arrow_compare, MaybeCompareFn, Operator};
use crate::stats::{ArrayStatistics, Stat};
use crate::ArrayData;

impl MaybeCompareFn for PrimitiveArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        // If the RHS is constant, then delegate to Arrow since.
        // TODO(ngates): remove these dual checks once we make stats not a hashmap
        //   https://github.com/spiraldb/vortex/issues/1309
        if ConstantArray::try_from(other).is_ok()
            || other
                .statistics()
                .get_as::<bool>(Stat::IsConstant)
                .unwrap_or(false)
        {
            return Some(arrow_compare(self.as_ref(), other, operator));
        }

        // If the RHS is primitive, then delegate to Arrow.
        if let Ok(primitive) = PrimitiveArray::try_from(other) {
            return Some(arrow_compare(self.as_ref(), primitive.as_ref(), operator));
        }

        None
    }
}
