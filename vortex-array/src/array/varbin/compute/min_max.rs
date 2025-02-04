use itertools::Itertools;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::accessor::ArrayAccessor;
use crate::array::{VarBinArray, VarBinEncoding};
use crate::compute::{MinMaxFn, MinMaxResult};

impl MinMaxFn<VarBinArray> for VarBinEncoding {
    fn min_max(&self, array: &VarBinArray) -> VortexResult<Option<MinMaxResult>> {
        compute_min_max(array, array.0.dtype())
    }
}

/// Compute the min and max of VarBin like array.
pub fn compute_min_max<T: ArrayAccessor<[u8]>>(
    array: &T,
    dtype: &DType,
) -> VortexResult<Option<MinMaxResult>> {
    let dtype = dtype.with_nullability(NonNullable);
    let minmax = array.with_iterator(|iter| match iter.flatten().minmax() {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(value) => {
            let scalar = Scalar::new(dtype, value.into());
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
            min: Scalar::new(dtype.clone(), (*min).into()),
            max: Scalar::new(dtype, (*max).into()),
        }),
    })?;

    Ok(minmax)
}

#[cfg(test)]
mod tests {

    use vortex_buffer::BufferString;
    use vortex_dtype::DType;

    use crate::array::varbin::Nullability;
    use crate::array::VarBinArray;
    use crate::compute::{min_max, MinMaxResult};
    use crate::stats::Stat;

    #[test]
    fn some_nulls() {
        let array = VarBinArray::from_iter(
            vec![
                Some("hello world"),
                None,
                Some("hello world this is a long string"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        );
        let MinMaxResult { min, max } = min_max(array).unwrap().unwrap();

        assert_eq!(min, BufferString::from("hello world".to_string()).into());
        assert_eq!(
            max,
            BufferString::from("hello world this is a long string".to_string()).into(),
        );
    }

    #[test]
    fn all_nulls() {
        let array = VarBinArray::from_iter(
            vec![Option::<&str>::None, None, None],
            DType::Utf8(Nullability::Nullable),
        );
        assert!(array.statistics().get(Stat::Min).is_none());
        assert!(array.statistics().get(Stat::Max).is_none());
    }
}
