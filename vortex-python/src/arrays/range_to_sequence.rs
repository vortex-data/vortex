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
use vortex::dtype::PType;
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

    // A `Sequence` holds at least one element, so an empty range returns a primitive array
    // instead. `range_len` reports a range that runs the wrong way as `None` and one whose bounds
    // coincide as `Some(0)`; Python's `range` calls both empty.
    let len = range_len(start, stop, step).unwrap_or(0);
    if len == 0 {
        let validity = match dtype.nullability() {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        return Ok(PrimitiveArray::new::<T>(Buffer::empty(), validity).into_array());
    }
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

/// The [`PType`] a Python `range` converts to when the caller does not request a dtype.
///
/// An unsigned type needs a positive step as well as positive bounds, because the step is stored
/// in the same type as the values.
pub fn range_ptype(start: isize, stop: isize, step: isize) -> PType {
    if start > 0 && stop > 0 && step > 0 {
        PType::U64
    } else {
        PType::I64
    }
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
        assert_ne!(step, 0);

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
    use rstest::rstest;
    use vortex::array::IntoArray as _;
    use vortex::array::assert_arrays_eq;
    use vortex::array::match_each_integer_ptype;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;

    use crate::arrays::range_to_sequence::range_len;
    use crate::arrays::range_to_sequence::range_ptype;
    use crate::arrays::range_to_sequence::sequence_array_from_range;

    /// Python's `range` is empty in three shapes, and `range_len` reports the first as `Some(0)`
    /// and the other two as `None`. All three must convert to an empty array.
    #[rstest]
    #[case::bounds_coincide(3, 3, 1)]
    #[case::bounds_coincide_below_zero(-5, -5, 1)]
    #[case::bounds_coincide_with_negative_step(3, 3, -1)]
    #[case::positive_step_runs_backwards(10, 3, 1)]
    #[case::negative_step_runs_forwards(0, 10, -1)]
    fn empty_range_converts_to_an_empty_array(
        #[case] start: isize,
        #[case] stop: isize,
        #[case] step: isize,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i64>(start, stop, step, dtype)?;
        assert_eq!(arr.len(), 0);
        Ok(())
    }

    /// A descending range needs a signed type, because the negative step is stored in the same
    /// type as the values.
    #[rstest]
    #[case::ascending_above_zero(1, 10, 1, PType::U64)]
    #[case::ascending_from_zero(0, 10, 1, PType::I64)]
    #[case::descending_above_zero(5, 1, -1, PType::I64)]
    #[case::descending_through_zero(5, -1, -1, PType::I64)]
    fn range_ptype_holds_the_step(
        #[case] start: isize,
        #[case] stop: isize,
        #[case] step: isize,
        #[case] expected: PType,
    ) {
        assert_eq!(range_ptype(start, stop, step), expected);
    }

    #[test]
    fn descending_range_above_zero_converts() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let ptype = range_ptype(5, 1, -1);
        let dtype = DType::Primitive(ptype, Nullability::NonNullable);
        let arr = match_each_integer_ptype!(ptype, |T| {
            sequence_array_from_range::<T>(5, 1, -1, dtype)
        })?;
        assert_arrays_eq!(arr, buffer![5i64, 4, 3, 2].into_array(), &mut ctx);
        Ok(())
    }

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
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::U16, Nullability::NonNullable);
        let arr = sequence_array_from_range::<u16>(0, 10, 1, dtype).unwrap();
        assert_arrays_eq!(
            arr,
            buffer![0u16, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            &mut ctx
        );

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i32>(0, 10, 5, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![0i32, 5].into_array(), &mut ctx);

        let dtype = DType::Primitive(PType::I8, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i8>(-5, 5, 3, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![-5i8, -2, 1, 4].into_array(), &mut ctx);

        let dtype = DType::Primitive(PType::I8, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i8>(3, -3, -1, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![3i8, 2, 1, 0, -1, -2].into_array(), &mut ctx);

        let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        let result = sequence_array_from_range::<u32>(1_000_000, 10, -500_000, dtype);
        assert!(
            result.is_err_and(|err| err.to_string().contains("does not fit in requested dtype"))
        );

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let arr = sequence_array_from_range::<i32>(1_000_000, 10, -500_000, dtype).unwrap();
        assert_arrays_eq!(arr, buffer![1_000_000i32, 500_000].into_array(), &mut ctx);
    }
}
