// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::iter::trimmed_ends_iter;

/// Run-end encode a `PrimitiveArray`, returning a tuple of `(ends, values)`.
pub fn runend_encode(
    array: ArrayView<Primitive>,
    ctx: &mut ExecutionCtx,
) -> (PrimitiveArray, ArrayRef) {
    let validity = match array
        .validity()
        .vortex_expect("run-end validity should be derivable")
    {
        Validity::NonNullable => None,
        Validity::AllValid => None,
        Validity::AllInvalid => {
            // We can trivially return an all-null REE array
            let ends = PrimitiveArray::new(buffer![array.len() as u64], Validity::NonNullable);
            ends.statistics()
                .set(Stat::IsStrictSorted, Precision::Exact(true.into()));
            return (
                ends,
                ConstantArray::new(Scalar::null(array.dtype().clone()), 1).into_array(),
            );
        }
        Validity::Array(a) => {
            let bool_array = a
                .execute::<BoolArray>(ctx)
                .vortex_expect("validity array must be convertible to bool");
            Some(bool_array.to_bit_buffer())
        }
    };

    let (ends, values) = match validity {
        None => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) = runend_encode_primitive(array.as_slice::<P>());
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    PrimitiveArray::new(values, array.dtype().nullability().into()).into_array(),
                )
            })
        }
        Some(validity) => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) =
                    runend_encode_nullable_primitive(array.as_slice::<P>(), validity);
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    values.into_array(),
                )
            })
        }
    };

    let ends = ends.narrow().vortex_expect("Ends must succeed downcasting");

    ends.statistics()
        .set(Stat::IsStrictSorted, Precision::Exact(true.into()));

    (ends, values)
}

fn runend_encode_primitive<T: NativePType>(elements: &[T]) -> (Buffer<u64>, Buffer<T>) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();

    if elements.is_empty() {
        return (ends.freeze(), values.freeze());
    }

    // Run-end encode the values
    let mut prev = elements[0];
    let mut end = 1;
    for &e in elements.iter().skip(1) {
        if e != prev {
            ends.push(end);
            values.push(prev);
        }
        prev = e;
        end += 1;
    }
    ends.push(end);
    values.push(prev);

    (ends.freeze(), values.freeze())
}

fn runend_encode_nullable_primitive<T: NativePType>(
    elements: &[T],
    element_validity: BitBuffer,
) -> (Buffer<u64>, PrimitiveArray) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();
    let mut validity = BitBufferMut::with_capacity(values.capacity());

    if elements.is_empty() {
        return (
            ends.freeze(),
            PrimitiveArray::new(
                values,
                Validity::Array(BoolArray::from(validity.freeze()).into_array()),
            ),
        );
    }

    // Run-end encode the values
    let mut prev = element_validity.value(0).then(|| elements[0]);
    let mut end = 1;
    for e in elements
        .iter()
        .zip(element_validity.iter())
        .map(|(&e, is_valid)| is_valid.then_some(e))
        .skip(1)
    {
        if e != prev {
            ends.push(end);
            match prev {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(p) => {
                    validity.append(true);
                    values.push(p);
                }
            }
        }
        prev = e;
        end += 1;
    }
    ends.push(end);

    match prev {
        None => {
            validity.append(false);
            values.push(T::default());
        }
        Some(p) => {
            validity.append(true);
            values.push(p);
        }
    }

    (
        ends.freeze(),
        PrimitiveArray::new(values, Validity::from(validity.freeze())),
    )
}

pub fn runend_decode_primitive(
    ends: PrimitiveArray,
    values: PrimitiveArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity_mask = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    Ok(match_each_native_ptype!(values.ptype(), |P| {
        match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            runend_decode_typed_primitive(
                trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
                values.as_slice::<P>(),
                validity_mask,
                values.dtype().nullability(),
                length,
            )
        })
    }))
}

/// Decode a run-end encoded slice of values into a flat `Buffer<T>` and `Validity`.
///
/// This is the core decode loop shared by primitive and varbinview run-end decoding.
fn runend_decode_slice<T: Copy + Default>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> (Buffer<T>, Validity) {
    match values_validity {
        Mask::AllTrue(_) => {
            let mut decoded: BufferMut<T> = BufferMut::with_capacity(length);
            for (end, value) in run_ends.zip_eq(values) {
                assert!(
                    end >= decoded.len(),
                    "Runend ends must be monotonic, got {end} after {}",
                    decoded.len()
                );
                assert!(end <= length, "Runend end must be less than overall length");
                // SAFETY:
                // We preallocate enough capacity because we know the total length
                unsafe { decoded.push_n_unchecked(*value, end - decoded.len()) };
            }
            (decoded.into(), values_nullability.into())
        }
        Mask::AllFalse(_) => (Buffer::<T>::zeroed(length), Validity::AllInvalid),
        Mask::Values(mask) => {
            let mut decoded = BufferMut::with_capacity(length);
            let mut decoded_validity = BitBufferMut::with_capacity(length);
            for (end, value) in run_ends.zip_eq(
                values
                    .iter()
                    .zip(mask.bit_buffer().iter())
                    .map(|(&v, is_valid)| is_valid.then_some(v)),
            ) {
                assert!(
                    end >= decoded.len(),
                    "Runend ends must be monotonic, got {end} after {}",
                    decoded.len()
                );
                assert!(end <= length, "Runend end must be less than overall length");
                match value {
                    None => {
                        decoded_validity.append_n(false, end - decoded.len());
                        // SAFETY:
                        // We preallocate enough capacity because we know the total length
                        unsafe { decoded.push_n_unchecked(T::default(), end - decoded.len()) };
                    }
                    Some(value) => {
                        decoded_validity.append_n(true, end - decoded.len());
                        // SAFETY:
                        // We preallocate enough capacity because we know the total length
                        unsafe { decoded.push_n_unchecked(value, end - decoded.len()) };
                    }
                }
            }
            (decoded.into(), Validity::from(decoded_validity.freeze()))
        }
    }
}

pub fn runend_decode_typed_primitive<T: NativePType>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> PrimitiveArray {
    let (decoded, validity) = runend_decode_slice(
        run_ends,
        values,
        values_validity,
        values_nullability,
        length,
    );
    PrimitiveArray::new(decoded, validity)
}

/// Decode a run-end encoded VarBinView array by expanding views directly.
pub fn runend_decode_varbinview(
    ends: PrimitiveArray,
    values: VarBinViewArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinViewArray> {
    let validity_mask = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    let views = values.views();

    let (decoded_views, validity) = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_slice(
            trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
            views,
            validity_mask,
            values.dtype().nullability(),
            length,
        )
    });

    let parts = values.into_data_parts();
    let view_handle = BufferHandle::new_host(decoded_views.into_byte_buffer());

    // SAFETY: we are expanding views from a valid VarBinViewArray with the same
    // buffers, so all buffer indices and offsets remain valid.
    Ok(unsafe {
        VarBinViewArray::new_handle_unchecked(view_handle, parts.buffers, parts.dtype, validity)
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::compress::runend_decode_primitive;
    use crate::compress::runend_encode;

    #[test]
    fn encode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = PrimitiveArray::from_iter([1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);
        let values = values.execute::<PrimitiveArray>(&mut ctx)?;

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 5, 10]);
        assert_arrays_eq!(ends, expected_ends);
        let expected_values = PrimitiveArray::from_iter(vec![1i32, 2, 3]);
        assert_arrays_eq!(values, expected_values);
        Ok(())
    }

    #[test]
    fn encode_nullable() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            Validity::from(BitBuffer::from(vec![
                true, true, false, false, true, true, true, true, false, false,
            ])),
        );
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);
        let values = values.execute::<PrimitiveArray>(&mut ctx)?;

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 4, 5, 8, 10]);
        assert_arrays_eq!(ends, expected_ends);
        let expected_values =
            PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(2), Some(3), None]);
        assert_arrays_eq!(values, expected_values);
        Ok(())
    }

    #[test]
    fn encode_all_null() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![0, 0, 0, 0, 0],
            Validity::from(BitBuffer::new_unset(5)),
        );
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);
        let values = values.execute::<PrimitiveArray>(&mut ctx)?;

        let expected_ends = PrimitiveArray::from_iter(vec![5u64]);
        assert_arrays_eq!(ends, expected_ends);
        let expected_values = PrimitiveArray::from_option_iter(vec![Option::<i32>::None]);
        assert_arrays_eq!(values, expected_values);
        Ok(())
    }

    #[test]
    fn decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 10, &mut ctx)?;

        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }
}
