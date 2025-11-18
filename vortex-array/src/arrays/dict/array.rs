// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_buffer::{BitBuffer, ByteBuffer};
use vortex_dtype::{DType, Nullability, PType, match_each_integer_ptype};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_err};
use vortex_mask::{AllOr, Mask};

use crate::builders::dict::dict_encode;
use crate::serde::ArrayChildren;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{ArrayVTable, EncodeVTable, NotSupported, VTable, ValidityVTable, VisitorVTable};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef, Canonical,
    DeserializeMetadata, EncodingId, EncodingRef, Precision, ProstMetadata, SerializeMetadata,
    ToCanonical, vtable,
};

vtable!(Dict);

#[derive(Clone, prost::Message)]
pub struct DictMetadata {
    #[prost(uint32, tag = "1")]
    pub(super) values_len: u32,
    #[prost(enumeration = "PType", tag = "2")]
    pub(super) codes_ptype: i32,
    // nullable codes are optional since they were added after stabilisation
    #[prost(optional, bool, tag = "3")]
    pub(super) is_nullable_codes: Option<bool>,
}

impl VTable for DictVTable {
    type Array = DictArray;
    type Encoding = DictEncoding;
    type Metadata = ProstMetadata<DictMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.dict")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(DictEncoding.as_ref())
    }

    fn metadata(array: &DictArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DictMetadata {
            codes_ptype: PType::try_from(array.codes().dtype())? as i32,
            values_len: u32::try_from(array.values().len()).map_err(|_| {
                vortex_err!(
                    "Dictionary values size {} overflowed u32",
                    array.values().len()
                )
            })?,
            is_nullable_codes: Some(array.codes().dtype().is_nullable()),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(buffer)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        _encoding: &DictEncoding,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DictArray> {
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // If no `is_nullable_codes` metadata use the nullability of the values
            // (and whole array) as before.
            .unwrap_or_else(|| dtype.nullability());
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;

        DictArray::try_new(codes, values)
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

impl EncodeVTable<DictVTable> for DictVTable {
    fn encode(
        _encoding: &DictEncoding,
        canonical: &Canonical,
        _like: Option<&DictArray>,
    ) -> VortexResult<Option<DictArray>> {
        Ok(Some(dict_encode(canonical.as_ref())?))
    }
}

impl VisitorVTable<DictVTable> for DictVTable {
    fn visit_buffers(_array: &DictArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DictArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", array.codes());
        visitor.visit_child("values", array.values());
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        use super::DictMetadata;
        use crate::ProstMetadata;
        use crate::test_harness::check_metadata;

        check_metadata(
            "dict.metadata",
            ProstMetadata(DictMetadata {
                codes_ptype: PType::U64 as i32,
                values_len: u32::MAX,
                is_nullable_codes: None,
            }),
        );
    }
}
