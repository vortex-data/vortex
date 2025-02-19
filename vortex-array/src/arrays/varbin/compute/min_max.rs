use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinArray, VarBinEncoding};
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
    let minmax = array.with_iterator(|iter| match iter.flatten().minmax() {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(value) => {
            let scalar = Scalar::new(dtype.clone(), value.into());
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
            min: Scalar::new(dtype.clone(), (*min).into()),
            max: Scalar::new(dtype.clone(), (*max).into()),
        }),
    })?;

    Ok(minmax)
}

#[cfg(test)]
mod tests {

    use vortex_buffer::BufferString;
    use vortex_dtype::DType::Utf8;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::arrays::VarBinArray;
    use crate::compute::{min_max, MinMaxResult};
    use crate::stats::{Stat, Statistics};

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
        let MinMaxResult { min, max } = min_max(array).unwrap().unwrap();

        assert_eq!(
            min,
            Scalar::new(
                Utf8(Nullable),
                BufferString::from("hello world".to_string()).into(),
            )
        );
        assert_eq!(
            max,
            Scalar::new(
                Utf8(Nullable),
                BufferString::from("hello world this is a long string".to_string()).into()
            )
        );
    }

    #[test]
    fn all_nulls() {
        let array = VarBinArray::from_iter(vec![Option::<&str>::None, None, None], Utf8(Nullable));
        assert!(array.get_stat(Stat::Min).is_none());
        assert!(array.get_stat(Stat::Max).is_none());
    }
}
