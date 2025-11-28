// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_scalar::Scalar;

use crate::accessor::ArrayAccessor;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

impl MinMaxKernel for VarBinVTable {
    fn min_max(&self, array: &VarBinArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(varbin_compute_min_max(array, array.dtype()))
    }
}

register_kernel!(MinMaxKernelAdapter(VarBinVTable).lift());

/// Compute the min and max of VarBin like array.
pub(crate) fn varbin_compute_min_max<T: ArrayAccessor<[u8]>>(
    array: &T,
    dtype: &DType,
) -> Option<MinMaxResult> {
    array.with_iterator(|iter| match iter.flatten().minmax() {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(value) => {
            let scalar = make_scalar(dtype, value);
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
            min: make_scalar(dtype, min),
            max: make_scalar(dtype, max),
        }),
    })
}

/// Helper function to make sure that min/max has the right [`Scalar`] type.
fn make_scalar(dtype: &DType, value: &[u8]) -> Scalar {
    match dtype {
        DType::Binary(_) => Scalar::binary(value.to_vec(), NonNullable),
        DType::Utf8(_) => {
            // SAFETY: We only call `compute_min_max` within `varbin/`, in which we always validate
            // the arrays, and we always pass `array.dtype()` in as the `dtype` argument.
            let value = unsafe { str::from_utf8_unchecked(value) };
            Scalar::utf8(value, NonNullable)
        }
        _ => vortex_panic!("cannot make Scalar from bytes with dtype {dtype}"),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferString;
    use vortex_dtype::DType::Utf8;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::arrays::VarBinArray;
    use crate::compute::MinMaxResult;
    use crate::compute::min_max;
    use crate::expr::stats::Stat;
    use crate::stats::StatsProvider;

    #[test]
    fn some_nulls() {
        let array = VarBinArray::from_iter(
            vec![
                Some("hello world"),
                None,
                Some("hello world this is a long string"),
                None,
            ],
            Utf8(Nullable),
        );
        let MinMaxResult { min, max } = min_max(array.as_ref()).unwrap().unwrap();

        assert_eq!(
            min,
            Scalar::new(
                Utf8(NonNullable),
                BufferString::from("hello world".to_string()).into(),
            )
        );
        assert_eq!(
            max,
            Scalar::new(
                Utf8(NonNullable),
                BufferString::from("hello world this is a long string".to_string()).into()
            )
        );
    }

    #[test]
    fn all_nulls() {
        let array = VarBinArray::from_iter(vec![Option::<&str>::None, None, None], Utf8(Nullable));
        let stats = array.statistics();
        assert!(stats.get(Stat::Min).is_none());
        assert!(stats.get(Stat::Max).is_none());
    }
}
