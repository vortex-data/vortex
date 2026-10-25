// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::slice::inherit_selection_stats;
use super::slice::slice_constant_code;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for Dict {
    fn filter(array: ArrayView<'_, Dict>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        if let Some(code) = array.codes().as_opt::<Constant>() {
            return slice_constant_code(array, code.scalar(), mask.true_count());
        }

        let codes = array.codes().filter(mask.clone())?;
        if let Some(code) = codes.as_opt::<Constant>() {
            return slice_constant_code(array, code.scalar(), codes.len());
        }
        inherit_selection_stats(array.codes(), &codes);

        // SAFETY: Selection preserves the integer code type and non-null code bounds, so every code
        // indexes the unchanged values. `new_unchecked` resets `all_values_referenced` because
        // selection can remove a value's last reference.
        let filtered = unsafe { DictArray::new_unchecked(codes, array.values().clone()) };

        Ok(Some(filtered.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::ScalarValue;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn filter_constant_valid_code_keeps_only_referenced_value() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::new(
            ConstantArray::new(1u8, 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let filtered = dict
            .filter(Mask::from_iter([true, false, true, true, false]))?
            .optimize(&SESSION)?;

        let filtered_dict = filtered.as_::<Dict>();
        assert_arrays_eq!(
            filtered_dict.values(),
            buffer![20i32].into_array(),
            &mut ctx
        );
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter([20i32, 20, 20]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn filter_inherits_true_order_stats_from_codes() -> VortexResult<()> {
        let codes = buffer![0u8, 0, 1, 2].into_array();
        codes
            .statistics()
            .set(Stat::IsSorted, Precision::exact(ScalarValue::Bool(true)));
        codes
            .statistics()
            .set(Stat::IsConstant, Precision::exact(ScalarValue::Bool(false)));
        let dict = DictArray::new(codes, buffer![10i32, 20, 30].into_array());

        let filtered = dict
            .filter(Mask::from_iter([true, false, true, true]))?
            .optimize(&SESSION)?;

        let filtered_codes = filtered.as_::<Dict>().codes().clone();
        assert_eq!(
            filtered_codes.statistics().get_as::<bool>(Stat::IsSorted),
            Precision::exact(true)
        );
        assert_eq!(
            filtered_codes.statistics().get_as::<bool>(Stat::IsConstant),
            Precision::Absent
        );
        Ok(())
    }
}
