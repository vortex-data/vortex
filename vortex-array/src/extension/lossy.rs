// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `Lossy` extension type for marking columns as "may be stored lossily".
//!
//! Wrapping a column with `Lossy` advertises that it is acceptable for compressors to use
//! lossy encodings (e.g. quantization). Only float-shaped storage is allowed: floating-point
//! primitives, lists / fixed-size lists thereof, and extension types that explicitly opt in
//! via [`ExtVTable::can_be_lossy`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::Matcher;
use crate::extension::EmptyMetadata;
use crate::scalar::ScalarValue;

/// The `Lossy` extension type, marking a column as lossy-storage-eligible.
///
/// `Lossy` wraps a storage dtype that recursively bottoms out in floating-point primitives,
/// optionally through `List`/`FixedSizeList` layers or other extension types whose
/// [`ExtVTable::can_be_lossy`] returns `true`.
///
/// # Examples
///
/// ```ignore
/// use vortex_array::dtype::{DType, Nullability, PType};
/// use vortex_array::extension::Lossy;
///
/// let lossy = Lossy::new(DType::Primitive(PType::F32, Nullability::NonNullable));
/// assert!(lossy.is_ok());
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Lossy;

impl Lossy {
    /// Creates a new `Lossy` extension dtype wrapping the given storage dtype.
    ///
    /// The storage type must recursively bottom out in floating-point primitives, optionally
    /// through `List` / `FixedSizeList` layers or extension types whose
    /// [`ExtVTable::can_be_lossy`] returns `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if `storage` is not a float-shaped dtype.
    pub fn new(storage: DType) -> VortexResult<ExtDType<Self>> {
        ExtDType::try_new(EmptyMetadata, storage)
    }
}

/// Recursively validates that `dtype` is a permissible storage type for [`Lossy`].
///
/// The rules are:
///
/// - `Primitive(p, _)` is allowed iff `p.is_float()`.
/// - `List(inner, _)` and `FixedSizeList(inner, _, _)` recurse into `inner`.
/// - `Extension(ext)` is allowed iff `ext.can_be_lossy()` returns `true`, and recursively
///   the storage dtype of `ext` is also a permissible `Lossy` storage type.
/// - Anything else is rejected.
fn check_lossy_storage(dtype: &DType) -> VortexResult<()> {
    match dtype {
        DType::Primitive(ptype, _) if ptype.is_float() => Ok(()),
        DType::List(inner, _) => check_lossy_storage(inner),
        DType::FixedSizeList(inner, ..) => check_lossy_storage(inner),
        DType::Extension(ext) if ext.can_be_lossy() => check_lossy_storage(ext.storage_dtype()),
        _ => vortex_bail!("Lossy storage dtype must be float-shaped, got {dtype}"),
    }
}

impl ExtVTable for Lossy {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.lossy")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        check_lossy_storage(ext_dtype.storage_dtype())
    }

    /// `Lossy` itself is never lossy-eligible: nesting `Lossy<Lossy<...>>` is forbidden.
    fn can_be_lossy(&self) -> bool {
        false
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}

/// Matcher for the [`Lossy`] extension type.
pub struct AnyLossy;

impl Matcher for AnyLossy {
    type Match<'a> = ();

    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        item.metadata_opt::<Lossy>().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::extension::lossy::Lossy;

    fn primitive(ptype: PType) -> DType {
        DType::Primitive(ptype, NonNullable)
    }

    fn list_of(inner: DType) -> DType {
        DType::List(Arc::new(inner), NonNullable)
    }

    fn fsl(inner: DType, size: u32) -> DType {
        DType::FixedSizeList(Arc::new(inner), size, NonNullable)
    }

    #[rstest]
    #[case::f16(PType::F16)]
    #[case::f32(PType::F32)]
    #[case::f64(PType::F64)]
    fn accepts_float_primitive(#[case] ptype: PType) -> VortexResult<()> {
        Lossy::new(primitive(ptype))?;
        Ok(())
    }

    #[rstest]
    #[case::nullable(Nullable)]
    #[case::non_nullable(NonNullable)]
    fn accepts_float_with_any_nullability(#[case] nullability: Nullability) -> VortexResult<()> {
        Lossy::new(DType::Primitive(PType::F32, nullability))?;
        Ok(())
    }

    #[test]
    fn accepts_list_of_float() -> VortexResult<()> {
        Lossy::new(list_of(primitive(PType::F32)))?;
        Ok(())
    }

    #[test]
    fn accepts_nested_list_of_float() -> VortexResult<()> {
        Lossy::new(list_of(list_of(primitive(PType::F32))))?;
        Ok(())
    }

    #[test]
    fn accepts_fixed_size_list_of_float() -> VortexResult<()> {
        Lossy::new(fsl(primitive(PType::F32), 4))?;
        Ok(())
    }

    #[rstest]
    #[case::i8(PType::I8)]
    #[case::i16(PType::I16)]
    #[case::i32(PType::I32)]
    #[case::i64(PType::I64)]
    #[case::u8(PType::U8)]
    #[case::u16(PType::U16)]
    #[case::u32(PType::U32)]
    #[case::u64(PType::U64)]
    fn rejects_integer_primitive(#[case] ptype: PType) {
        assert!(Lossy::new(primitive(ptype)).is_err());
    }

    #[test]
    fn rejects_bool() {
        assert!(Lossy::new(DType::Bool(NonNullable)).is_err());
    }

    #[test]
    fn rejects_utf8() {
        assert!(Lossy::new(DType::Utf8(NonNullable)).is_err());
    }

    #[test]
    fn rejects_binary() {
        assert!(Lossy::new(DType::Binary(NonNullable)).is_err());
    }

    #[test]
    fn rejects_struct_of_float() {
        let fields = StructFields::from_iter([("a", primitive(PType::F32))]);
        assert!(Lossy::new(DType::Struct(fields, NonNullable)).is_err());
    }

    #[test]
    fn rejects_list_of_struct() {
        let fields = StructFields::from_iter([("a", primitive(PType::F32))]);
        let struct_dtype = DType::Struct(fields, NonNullable);
        assert!(Lossy::new(list_of(struct_dtype)).is_err());
    }

    #[test]
    fn rejects_nested_lossy() -> VortexResult<()> {
        let inner = DType::Extension(Lossy::new(primitive(PType::F32))?.erased());
        assert!(Lossy::new(inner).is_err());
        Ok(())
    }

    #[test]
    fn rejects_timestamp() {
        // Timestamp's `can_be_lossy` defaults to false.
        use crate::extension::datetime::TimeUnit;
        use crate::extension::datetime::Timestamp;
        let ts_dtype = DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullable).erased());
        assert!(Lossy::new(ts_dtype).is_err());
    }

    #[test]
    fn ext_dtype_reports_can_be_lossy_false() -> VortexResult<()> {
        // The Lossy ext type itself is not lossy-eligible.
        let ext = Lossy::new(primitive(PType::F32))?;
        assert!(!ext.can_be_lossy());
        Ok(())
    }

    #[test]
    fn ext_dtype_ref_reports_can_be_lossy_false() -> VortexResult<()> {
        let ext = Lossy::new(primitive(PType::F32))?.erased();
        assert!(!ext.can_be_lossy());
        Ok(())
    }

    #[test]
    fn timestamp_default_can_be_lossy_false() {
        use crate::extension::datetime::TimeUnit;
        use crate::extension::datetime::Timestamp;
        let ts = Timestamp::new(TimeUnit::Seconds, Nullable);
        assert!(!ts.can_be_lossy());
    }

    #[test]
    fn matcher_matches_lossy() -> VortexResult<()> {
        use crate::dtype::extension::Matcher;
        use crate::extension::lossy::AnyLossy;

        let lossy = Lossy::new(primitive(PType::F32))?.erased();
        assert!(AnyLossy::matches(&lossy));
        Ok(())
    }

    #[test]
    fn matcher_does_not_match_other_extension() {
        use crate::dtype::extension::Matcher;
        use crate::extension::datetime::TimeUnit;
        use crate::extension::datetime::Timestamp;
        use crate::extension::lossy::AnyLossy;

        let ts = Timestamp::new(TimeUnit::Seconds, Nullable).erased();
        assert!(!AnyLossy::matches(&ts));
    }

    #[test]
    fn dtype_peel_lossy_unwraps() -> VortexResult<()> {
        let storage = primitive(PType::F32);
        let lossy_dtype = DType::Extension(Lossy::new(storage.clone())?.erased());
        assert_eq!(lossy_dtype.peel_lossy(), &storage);
        Ok(())
    }

    #[test]
    fn dtype_peel_lossy_returns_self_for_non_lossy() {
        let dtype = primitive(PType::F32);
        assert_eq!(dtype.peel_lossy(), &dtype);
    }

    #[test]
    fn dtype_peel_lossy_returns_self_for_other_extension() {
        use crate::extension::datetime::TimeUnit;
        use crate::extension::datetime::Timestamp;

        let ts_dtype = DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullable).erased());
        assert_eq!(ts_dtype.peel_lossy(), &ts_dtype);
    }
}
