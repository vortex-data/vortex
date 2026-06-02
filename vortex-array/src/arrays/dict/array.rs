// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::smallvec;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::LEGACY_SESSION;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::Dict;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_integer_ptype;

#[derive(Clone, prost::Message)]
pub struct DictMetadata {
    #[prost(uint32, tag = "1")]
    pub(super) values_len: u32,
    #[prost(enumeration = "PType", tag = "2")]
    pub(super) codes_ptype: i32,
    // nullable codes are optional since they were added after stabilisation.
    #[prost(optional, bool, tag = "3")]
    pub(super) is_nullable_codes: Option<bool>,
    // all_values_referenced is optional for backward compatibility.
    // true = all dictionary values are definitely referenced by at least one code.
    // false/None = unknown whether all values are referenced (conservative default).
    #[prost(optional, bool, tag = "4")]
    pub(super) all_values_referenced: Option<bool>,
}

#[array_slots(Dict)]
pub struct DictSlots {
    /// The codes array mapping each element to a dictionary entry.
    pub codes: ArrayRef,
    /// The dictionary values array containing the unique values.
    pub values: ArrayRef,
}

#[derive(Debug, Clone)]
pub struct DictData {
    /// `true` if every dictionary value is referenced by at least one code. An incorrect
    /// value here can cause incorrect query results (e.g. min/max) but never UB.
    pub(super) all_values_referenced: bool,
}

impl Display for DictData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "all_values_referenced: {}",
            self.all_values_referenced
        )
    }
}

impl DictData {
    /// Build a new `DictArray` without validating the codes or values.
    ///
    /// # Safety
    /// This should be called only when you can guarantee the invariants checked
    /// by the safe `DictArray::try_new` constructor are valid, for example when
    /// you are filtering or slicing an existing valid `DictArray`.
    pub unsafe fn new_unchecked() -> Self {
        Self {
            all_values_referenced: false,
        }
    }

    /// Set whether all dictionary values are referenced by at least one code.
    ///
    /// # Safety
    /// Setting `true` when some values are unreferenced does not cause UB but will produce
    /// incorrect results from operations like min/max.
    pub unsafe fn set_all_values_referenced(mut self, all_values_referenced: bool) -> Self {
        self.all_values_referenced = all_values_referenced;
        self
    }

    /// Build a new `DictArray` from its components, `codes` and `values`.
    ///
    /// This constructor will panic if `codes` or `values` do not pass validation for building
    /// a new `DictArray`. See `DictArray::try_new` for a description of the error conditions.
    pub fn new(codes_dtype: &DType) -> Self {
        Self::try_new(codes_dtype).vortex_expect("DictArray new")
    }

    /// Build a new `DictArray` from its components, `codes` and `values`.
    ///
    /// The codes must be integers, and may be nullable. Values can be any
    /// type, and may also be nullable. This mirrors the nullability of the Arrow `DictionaryArray`.
    ///
    /// # Errors
    ///
    /// The `codes` **must** be integers, and the maximum code must be less than the length
    /// of the `values` array. Otherwise, this constructor returns an error.
    ///
    /// It is an error to provide a nullable `codes` with non-nullable `values`.
    pub(crate) fn try_new(codes_dtype: &DType) -> VortexResult<Self> {
        if !codes_dtype.is_int() {
            vortex_bail!(MismatchedTypes: "int", codes_dtype);
        }

        Ok(unsafe { Self::new_unchecked() })
    }
}

pub trait DictArrayExt: TypedArrayRef<Dict> + DictArraySlotsExt {
    #[inline]
    fn has_all_values_referenced(&self) -> bool {
        self.all_values_referenced
    }

    /// Returns whether the dictionary values are sorted in ascending order. Sourced
    /// entirely from the values array's `Stat::IsSorted` (plus O(1) structural fallbacks
    /// — empty, single-element, or constant). Never scans values.
    ///
    /// `sort_dict` and the sorted-dict writer cache `Stat::IsSorted = true` on the values
    /// they produce; transforms that preserve the values array (take/filter/slice) reuse
    /// the same stats cache via Arc-cloning, so propagation is automatic.
    fn has_sorted_values(&self) -> bool {
        infer_sorted_values(self.values())
    }

    fn validate_all_values_referenced(&self) -> VortexResult<()> {
        if self.has_all_values_referenced() {
            if !self.codes().is_host() {
                return Ok(());
            }

            let referenced_mask = self.compute_referenced_values_mask(true)?;
            let all_referenced = referenced_mask.iter().all(|v| v);

            vortex_ensure!(all_referenced, "value in dict not referenced");
        }

        Ok(())
    }

    fn compute_referenced_values_mask(&self, referenced: bool) -> VortexResult<BitBuffer> {
        let codes = self.codes();
        let codes_validity = codes
            .validity()?
            .execute_mask(codes.len(), &mut LEGACY_SESSION.create_execution_ctx())?;
        #[expect(deprecated)]
        let codes_primitive = self.codes().to_primitive();
        let values_len = self.values().len();

        let init_value = !referenced;
        let referenced_value = referenced;

        let mut values_vec = vec![init_value; values_len];
        match codes_validity.bit_buffer() {
            AllOr::All => {
                match_each_integer_ptype!(codes_primitive.ptype(), |P| {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "codes are non-negative indices; a negative signed code would wrap to a large usize and panic on the bounds-checked array index"
                    )]
                    for &idx in codes_primitive.as_slice::<P>() {
                        values_vec[idx as usize] = referenced_value;
                    }
                });
            }
            AllOr::None => {}
            AllOr::Some(mask) => {
                match_each_integer_ptype!(codes_primitive.ptype(), |P| {
                    let codes = codes_primitive.as_slice::<P>();

                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "codes are non-negative indices; a negative signed code would wrap to a large usize and panic on the bounds-checked array index"
                    )]
                    mask.set_indices().for_each(|idx| {
                        values_vec[codes[idx] as usize] = referenced_value;
                    });
                });
            }
        }

        Ok(BitBuffer::from(values_vec))
    }
}
impl<T: TypedArrayRef<Dict>> DictArrayExt for T {}

/// O(1) inference of whether `values` is sorted, used when the explicit flag isn't set.
/// Checks only structural conditions and cached stats; never scans the values.
///
/// The dict reader wraps the materialized values in a `Shared` array for de-duplication;
/// look through that wrapper to read the underlying array's stats.
fn infer_sorted_values(values: &ArrayRef) -> bool {
    use crate::arrays::Constant;
    use crate::arrays::Shared;
    use crate::arrays::shared::SharedArrayExt;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;

    // The dict reader wraps materialized values in `Shared`; check the underlying
    // source inside the if-let so the typed view stays alive while we read stats.
    if let Some(shared) = values.as_opt::<Shared>() {
        let src = shared.source();
        if src.len() <= 1 || src.is::<Constant>() {
            return true;
        }
        return matches!(
            src.statistics().get_as::<bool>(Stat::IsSorted),
            Precision::Exact(true)
        );
    }

    if values.len() <= 1 || values.is::<Constant>() {
        return true;
    }
    matches!(
        values.statistics().get_as::<bool>(Stat::IsSorted),
        Precision::Exact(true)
    )
}

/// Concrete parts of a [`DictArray`](super::DictArray) after iterative execution.
pub struct DictParts {
    pub dtype: DType,
    pub codes: ArrayRef,
    pub values: ArrayRef,
}

pub trait DictOwnedExt {
    fn into_parts(self) -> DictParts;
}

impl DictOwnedExt for Array<Dict> {
    fn into_parts(self) -> DictParts {
        match self.try_into_parts() {
            Ok(array_parts) => {
                let slots = DictSlots::from_slots(array_parts.slots);
                DictParts {
                    dtype: array_parts.dtype,
                    codes: slots.codes,
                    values: slots.values,
                }
            }
            Err(array) => {
                let slots = DictSlotsView::from_slots(array.slots());
                DictParts {
                    dtype: array.dtype().clone(),
                    codes: slots.codes.clone(),
                    values: slots.values.clone(),
                }
            }
        }
    }
}

impl Array<Dict> {
    /// Build a new `DictArray` from its components, `codes` and `values`.
    pub fn new(codes: ArrayRef, values: ArrayRef) -> Self {
        Self::try_new(codes, values).vortex_expect("DictArray new")
    }

    /// Build a new `DictArray` from its components, `codes` and `values`.
    pub fn try_new(codes: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
        let dtype = values
            .dtype()
            .union_nullability(codes.dtype().nullability());
        let len = codes.len();
        let data = DictData::try_new(codes.dtype())?;
        Array::try_from_parts(
            ArrayParts::new(Dict, dtype, len, data)
                .with_slots(smallvec![Some(codes), Some(values)]),
        )
    }

    /// Build a new `DictArray` without validating the codes or values.
    ///
    /// # Safety
    ///
    /// See [`DictData::new_unchecked`].
    pub unsafe fn new_unchecked(codes: ArrayRef, values: ArrayRef) -> Self {
        let dtype = values
            .dtype()
            .union_nullability(codes.dtype().nullability());
        let len = codes.len();
        let data = unsafe { DictData::new_unchecked() };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Dict, dtype, len, data)
                    .with_slots(smallvec![Some(codes), Some(values)]),
            )
        }
    }

    /// Set whether all values in the dictionary are referenced by at least one code.
    ///
    /// # Safety
    ///
    /// See [`DictData::set_all_values_referenced`].
    pub unsafe fn set_all_values_referenced(self, all_values_referenced: bool) -> Self {
        let dtype = self.dtype().clone();
        let len = self.len();
        let slots: ArraySlots = self.slots().iter().cloned().collect();
        let data = unsafe {
            self.into_data()
                .set_all_values_referenced(all_values_referenced)
        };
        let array = unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Dict, dtype, len, data).with_slots(slots))
        };

        #[cfg(debug_assertions)]
        if all_values_referenced {
            array
                .validate_all_values_referenced()
                .vortex_expect("validation should succeed when all values are referenced");
        }

        array
    }

}

#[cfg(test)]
mod test {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::distr::Distribution;
    use rand::distr::StandardUniform;
    use rand::prelude::StdRng;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;
    use vortex_mask::AllOr;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::dtype::DType;
    use crate::dtype::NativePType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::UnsignedPType;
    use crate::validity::Validity;

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
        let mask = dict
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                dict.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();
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
        let mask = dict
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                dict.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();
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
        let mask = dict
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                dict.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();
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
        let mask = dict
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                dict.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();
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
                    .vortex_expect("DictArray creation should succeed in arbitrary impl")
                    .into_array()
            })
            .collect::<ChunkedArray>()
            .into_array()
    }

    #[test]
    fn test_dict_array_from_primitive_chunks() -> VortexResult<()> {
        let len = 2;
        let chunk_count = 2;
        let array = make_dict_primitive_chunks::<u64, u64>(len, 2, chunk_count);

        let mut builder = builder_with_capacity(
            &DType::Primitive(PType::U64, NonNullable),
            len * chunk_count,
        );
        array.append_to_builder(builder.as_mut(), &mut LEGACY_SESSION.create_execution_ctx())?;

        #[expect(deprecated)]
        let into_prim = array.to_primitive();
        let prim_into = builder.finish_into_canonical().into_primitive();

        assert_arrays_eq!(into_prim, prim_into);
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        use prost::Message;

        use super::DictMetadata;
        use crate::test_harness::check_metadata;

        check_metadata(
            "dict.metadata",
            &DictMetadata {
                codes_ptype: PType::U64 as i32,
                values_len: u32::MAX,
                is_nullable_codes: None,
                all_values_referenced: None,
            }
            .encode_to_vec(),
        );
    }

    #[test]
    fn sorted_values_inferred_for_constant_values() -> VortexResult<()> {
        use crate::arrays::ConstantArray;
        use crate::arrays::dict::DictArrayExt;
        // ConstantArray values: only one logical value, trivially sorted.
        let values = ConstantArray::new(42i32, 1).into_array();
        let codes = buffer![0u32, 0, 0].into_array();
        let dict = DictArray::try_new(codes, values)?;
        assert!(dict.has_sorted_values());
        Ok(())
    }

    #[test]
    fn sorted_values_inferred_from_is_sorted_stat() -> VortexResult<()> {
        use crate::arrays::dict::DictArrayExt;
        use crate::expr::stats::Precision;
        use crate::expr::stats::Stat;
        // Values are sorted in fact but the flag isn't set. Once the IsSorted stat is
        // cached on the values array, the dict's inference picks it up.
        let values = buffer![1i32, 2, 3].into_array();
        let codes = buffer![0u32, 1, 2, 1, 0].into_array();
        let dict = DictArray::try_new(codes, values.clone())?;
        // Before the stat is set, the bare flag is false and no other cheap signal
        // applies — inference returns false.
        assert!(!dict.has_sorted_values());
        values
            .statistics()
            .set(Stat::IsSorted, Precision::Exact(true.into()));
        assert!(dict.has_sorted_values());
        Ok(())
    }
}
