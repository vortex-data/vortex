// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_buffer::BitBuffer;
use vortex_dtype::{DType, match_each_integer_ptype};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};
use vortex_mask::{AllOr, Mask};

use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{ArrayVTable, NotSupported, VTable, ValidityVTable};
use crate::{
    Array, ArrayEq, ArrayHash, ArrayRef, EncodingId, EncodingRef, Precision, ToCanonical, vtable,
};

vtable!(Dict);

impl VTable for DictVTable {
    type Array = DictArray;
    type Encoding = DictEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.dict")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(DictEncoding.as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct DictArray {
    codes: ArrayRef,
    values: ArrayRef,
    stats_set: ArrayStats,
    dtype: DType,
}

#[derive(Clone, Debug)]
pub struct DictEncoding;

impl DictArray {
    /// Build a new `DictArray` without validating the codes or values.
    ///
    /// # Safety
    /// This should be called only when you can guarantee the invariants checked
    /// by the safe [`DictArray::try_new`] constructor are valid, for example when
    /// you are filtering or slicing an existing valid `DictArray`.
    pub unsafe fn new_unchecked(codes: ArrayRef, values: ArrayRef) -> Self {
        let dtype = values
            .dtype()
            .union_nullability(codes.dtype().nullability());
        Self {
            codes,
            values,
            stats_set: Default::default(),
            dtype,
        }
    }

    /// Build a new `DictArray` from its components, `codes` and `values`.
    ///
    /// This constructor will panic if `codes` or `values` do not pass validation for building
    /// a new `DictArray`. See [`DictArray::try_new`] for a description of the error conditions.
    pub fn new(codes: ArrayRef, values: ArrayRef) -> Self {
        Self::try_new(codes, values).vortex_expect("DictArray new")
    }

    /// Build a new `DictArray` from its components, `codes` and `values`.
    ///
    /// The codes must be unsigned integers, and may be nullable. Values can be any type, and
    /// may also be nullable. This mirrors the nullability of the Arrow `DictionaryArray`.
    ///
    /// # Errors
    ///
    /// The `codes` **must** be unsigned integers, and the maximum code must be less than the length
    /// of the `values` array. Otherwise, this constructor returns an error.
    ///
    /// It is an error to provide a nullable `codes` with non-nullable `values`.
    pub fn try_new(codes: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
        if !codes.dtype().is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", codes.dtype());
        }

        Ok(unsafe { Self::new_unchecked(codes, values) })
    }

    #[inline]
    pub fn codes(&self) -> &ArrayRef {
        &self.codes
    }

    #[inline]
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// Compute a mask indicating which values in the dictionary are referenced by at least one code.
    ///
    /// Returns a `BitBuffer` where unset bits (false) correspond to values that are referenced
    /// by at least one valid code, and set bits (true) correspond to unreferenced values.
    ///
    /// This is useful for operations like min/max that need to ignore unreferenced values.
    pub fn compute_unreferenced_values_mask(&self) -> VortexResult<BitBuffer> {
        let codes_validity = self.codes().validity_mask();
        let codes_primitive = self.codes().to_primitive();
        let values_len = self.values().len();

        let mut unreferenced_vec = vec![true; values_len];
        match codes_validity.bit_buffer() {
            AllOr::All => {
                match_each_integer_ptype!(codes_primitive.ptype(), |P| {
                    #[allow(clippy::cast_possible_truncation)]
                    for &code in codes_primitive.as_slice::<P>().iter() {
                        unreferenced_vec[code as usize] = false;
                    }
                });
            }
            AllOr::None => {}
            AllOr::Some(buf) => {
                match_each_integer_ptype!(codes_primitive.ptype(), |P| {
                    let codes = codes_primitive.as_slice::<P>();

                    #[allow(clippy::cast_possible_truncation)]
                    buf.set_indices().for_each(|idx| {
                        unreferenced_vec[codes[idx] as usize] = false;
                    })
                });
            }
        }

        Ok(BitBuffer::collect_bool(values_len, |idx| {
            unreferenced_vec[idx]
        }))
    }
}

impl ArrayVTable<DictVTable> for DictVTable {
    fn len(array: &DictArray) -> usize {
        array.codes.len()
    }

    fn dtype(array: &DictArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DictArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DictArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.codes.array_hash(state, precision);
        array.values.array_hash(state, precision);
    }

    fn array_eq(array: &DictArray, other: &DictArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.codes.array_eq(&other.codes, precision)
            && array.values.array_eq(&other.values, precision)
    }
}

impl ValidityVTable<DictVTable> for DictVTable {
    fn is_valid(array: &DictArray, index: usize) -> bool {
        let scalar = array.codes().scalar_at(index);

        if scalar.is_null() {
            return false;
        };
        let values_index: usize = scalar
            .as_ref()
            .try_into()
            .vortex_expect("Failed to convert dictionary code to usize");
        array.values().is_valid(values_index)
    }

    fn all_valid(array: &DictArray) -> bool {
        array.codes().all_valid() && array.values().all_valid()
    }

    fn all_invalid(array: &DictArray) -> bool {
        array.codes().all_invalid() || array.values().all_invalid()
    }

    fn validity_mask(array: &DictArray) -> Mask {
        let codes_validity = array.codes().validity_mask();
        match codes_validity.bit_buffer() {
            AllOr::All => {
                let primitive_codes = array.codes().to_primitive();
                let values_mask = array.values().validity_mask();
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |P| {
                    let codes_slice = primitive_codes.as_slice::<P>();
                    BitBuffer::collect_bool(array.len(), |idx| {
                        #[allow(clippy::cast_possible_truncation)]
                        values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Mask::from_buffer(is_valid_buffer)
            }
            AllOr::None => Mask::AllFalse(array.len()),
            AllOr::Some(validity_buff) => {
                let primitive_codes = array.codes().to_primitive();
                let values_mask = array.values().validity_mask();
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |P| {
                    let codes_slice = primitive_codes.as_slice::<P>();
                    #[allow(clippy::cast_possible_truncation)]
                    BitBuffer::collect_bool(array.len(), |idx| {
                        validity_buff.value(idx) && values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Mask::from_buffer(is_valid_buffer)
            }
        }
    }
}

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use itertools::Itertools;
    use rand::distr::{Distribution, StandardUniform};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_buffer::{BitBuffer, buffer};
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, NativePType, PType, UnsignedPType};
    use vortex_error::{VortexExpect, VortexUnwrap, vortex_panic};
    use vortex_mask::AllOr;

    use crate::arrays::dict::DictArray;
    use crate::arrays::{ChunkedArray, PrimitiveArray};
    use crate::builders::builder_with_capacity;
    use crate::validity::Validity;
    use crate::{Array, ArrayRef, IntoArray, ToCanonical, assert_arrays_eq};

    #[test]
    fn nullable_codes_validity() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2, 2, 1],
                Validity::from(BitBuffer::from(vec![true, false, true, false, true])),
            )
            .into_array(),
            PrimitiveArray::new(buffer![3, 6, 9], Validity::AllValid).into_array(),
        )
        .unwrap();
        let mask = dict.validity_mask();
        let AllOr::Some(indices) = mask.indices() else {
            vortex_panic!("Expected indices from mask")
        };
        assert_eq!(indices, [0, 2, 4]);
    }

    #[test]
    fn nullable_values_validity() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2, 2, 1].into_array(),
            PrimitiveArray::new(
                buffer![3, 6, 9],
                Validity::from(BitBuffer::from(vec![true, false, false])),
            )
            .into_array(),
        )
        .unwrap();
        let mask = dict.validity_mask();
        let AllOr::Some(indices) = mask.indices() else {
            vortex_panic!("Expected indices from mask")
        };
        assert_eq!(indices, [0]);
    }

    #[test]
    fn nullable_codes_and_values() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2, 2, 1],
                Validity::from(BitBuffer::from(vec![true, false, true, false, true])),
            )
            .into_array(),
            PrimitiveArray::new(
                buffer![3, 6, 9],
                Validity::from(BitBuffer::from(vec![false, true, true])),
            )
            .into_array(),
        )
        .unwrap();
        let mask = dict.validity_mask();
        let AllOr::Some(indices) = mask.indices() else {
            vortex_panic!("Expected indices from mask")
        };
        assert_eq!(indices, [2, 4]);
    }

    #[test]
    fn nullable_codes_and_non_null_values() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2, 2, 1],
                Validity::from(BitBuffer::from(vec![true, false, true, false, true])),
            )
            .into_array(),
            PrimitiveArray::new(buffer![3, 6, 9], Validity::NonNullable).into_array(),
        )
        .unwrap();
        let mask = dict.validity_mask();
        let AllOr::Some(indices) = mask.indices() else {
            vortex_panic!("Expected indices from mask")
        };
        assert_eq!(indices, [0, 2, 4]);
    }

    fn make_dict_primitive_chunks<T: NativePType, Code: UnsignedPType>(
        len: usize,
        unique_values: usize,
        chunk_count: usize,
    ) -> ArrayRef
    where
        StandardUniform: Distribution<T>,
    {
        let mut rng = StdRng::seed_from_u64(0);

        (0..chunk_count)
            .map(|_| {
                let values = (0..unique_values)
                    .map(|_| rng.random::<T>())
                    .collect::<PrimitiveArray>();
                let codes = (0..len)
                    .map(|_| {
                        Code::from(rng.random_range(0..unique_values)).vortex_expect("valid value")
                    })
                    .collect::<PrimitiveArray>();

                DictArray::try_new(codes.into_array(), values.into_array())
                    .vortex_unwrap()
                    .into_array()
            })
            .collect::<ChunkedArray>()
            .into_array()
    }

    #[test]
    fn test_dict_array_from_primitive_chunks() {
        let len = 2;
        let chunk_count = 2;
        let array = make_dict_primitive_chunks::<u64, u64>(len, 2, chunk_count);

        let mut builder = builder_with_capacity(
            &DType::Primitive(PType::U64, NonNullable),
            len * chunk_count,
        );
        array.clone().append_to_builder(builder.as_mut());

        let into_prim = array.to_primitive();
        let prim_into = builder.finish_into_canonical().into_primitive();

        assert_arrays_eq!(into_prim, prim_into);
    }
}
