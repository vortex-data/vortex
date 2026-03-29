// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::encodings::sequence::Sequence;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::scalar::PValue;

pub fn sequence_array_from_range<T: NativePType + TryFrom<isize> + Into<PValue>>(
    start: isize,
    stop: isize,
    step: isize,
    dtype: DType,
) -> VortexResult<ArrayRef> {
    if step == 0 {
        vortex_bail!("Step must not be zero");
    }

    let Some(len) = range_len(start, stop, step) else {
        let validity = match dtype.nullability() {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        return Ok(PrimitiveArray::new::<T>(Buffer::empty(), validity).into_array());
    };
    let Ok(start) = T::try_from(start) else {
        vortex_bail!(
            "Start, {}, does not fit in requested dtype: {}",
            start,
            dtype
        );
    };
    let Ok(step) = T::try_from(step) else {
        vortex_bail!("Step, {}, does not fit in requested dtype: {}", step, dtype);
    };

    Ok(Sequence::try_new_typed::<T>(start, step, dtype.nullability(), len)?.into_array())
}

fn range_len(start: isize, stop: isize, step: isize) -> Option<usize> {
    if step > 0 {
        if start > stop {
            return None;
        }

        let len = (stop - start + step - 1) / step;
        let len =
            usize::try_from(len).vortex_expect("stop >= start, step > 0, so len is non-negative");
        Some(len)
    } else {
        assert!(step != 0);

        if stop > start {
            return None;
        }

        let len = (start - stop + -step - 1) / -step;
        let len =
            usize::try_from(len).vortex_expect("start >= stop, step < 0, so len is non-negative");
        Some(len)
    }
}

#[cfg(test)]
mod test {
    use vortex::array::IntoArray as _;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;

    use crate::arrays::range_to_sequence::range_len;
    use crate::arrays::range_to_sequence::sequence_array_from_range;

    #[test]
    fn test_range_len() {
        assert_eq!(range_len(0, 10, 1).unwrap(), 10);
        assert_eq!(range_len(0, 10, 5).unwrap(), 2);
        assert_eq!(range_len(0, 10, 10).unwrap(), 1);
        assert_eq!(range_len(0, 10, 100).unwrap(), 1);
        assert_eq!(range_len(-5, -5, 1).unwrap(), 0);
        assert_eq!(range_len(-5, 5, 3).unwrap(), 4);
        assert_eq!(range_len(-7, -5, 1).unwrap(), 2);
        assert_eq!(range_len(3, -3, -1).unwrap(), 6);
        assert_eq!(range_len(10, 3, 1), None);
        assert_eq!(range_len(0, 10, -1), None);
    }

    #[test]
    fn test_sequence_array_from_len() {
        let dtype = DType::Primitive(PType::U16, Nullability::NonNullable);
        let arr = sequence_array_from_range::<u16>(0, 10, 1, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![0u16, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array());

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i32>(0, 10, 5, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![0i32, 5].into_array());

        let dtype = DType::Primitive(PType::I8, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i8>(-5, 5, 3, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![-5i8, -2, 1, 4].into_array());

        let dtype = DType::Primitive(PType::I8, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i8>(3, -3, -1, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![3i8, 2, 1, 0, -1, -2].into_array());

        let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        let result = sequence_array_from_range::<u32>(1_000_000, 10, -500_000, dtype);
        assert!(
            result.is_err_and(|err| err.to_string().contains("does not fit in requested dtype"))
        );

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i32>(1_000_000, 10, -500_000, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![1_000_000i32, 500_000].into_array());
    }
}
