// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression tests for extension dtype construction, reads, nullability changes, and debug
//! output.

use std::hash::DefaultHasher;
use std::hash::Hasher;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::ExtDType;
use super::ExtDTypePlugin;
use super::ExtDTypeRef;
use super::ExtId;
use super::ExtVTable;
use super::ForeignExtDType;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// The number of times [`CountingValidator::validate_metadata`] ran.
static VALIDATIONS: AtomicUsize = AtomicUsize::new(0);

/// The number of times [`CountingValidator::storage_dtype`] ran.
static DECLARATIONS: AtomicUsize = AtomicUsize::new(0);

/// An extension type over `i64` storage that counts how often its metadata is validated.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
struct CountingValidator;

impl ExtVTable for CountingValidator {
    type Metadata = u32;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.counting_validator");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.to_le_bytes().to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        let bytes = metadata
            .try_into()
            .map_err(|_| vortex_err!("Expected four metadata bytes, got {}", metadata.len()))?;

        Ok(u32::from_le_bytes(bytes))
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        assert!(storage_dtype.is_nullable());
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DECLARATIONS.fetch_add(1, Ordering::Relaxed);

        DType::Primitive(PType::I64, NonNullable)
    }

    fn validate_metadata(_metadata: &Self::Metadata) -> VortexResult<()> {
        VALIDATIONS.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}

// One test owns both counters, so test threads cannot race on them.
#[test]
fn test_nullability_changes_do_not_validate_again() -> VortexResult<()> {
    for initial in [NonNullable, Nullable] {
        VALIDATIONS.store(0, Ordering::Relaxed);
        DECLARATIONS.store(0, Ordering::Relaxed);
        let dtype = ExtDType::<CountingValidator>::try_new(42, initial)?;
        assert_eq!(VALIDATIONS.load(Ordering::Relaxed), 1);

        for target in [Nullable, NonNullable] {
            let changed = dtype.with_nullability(target);

            assert_eq!(
                changed.storage_dtype(),
                &DType::Primitive(PType::I64, target)
            );
            assert_eq!(changed.metadata::<CountingValidator>(), &42);
            assert_eq!(changed.id(), dtype.id());
        }

        assert_eq!(VALIDATIONS.load(Ordering::Relaxed), 1);
        assert_eq!(DECLARATIONS.load(Ordering::Relaxed), 3);

        VALIDATIONS.store(0, Ordering::Relaxed);
        DECLARATIONS.store(0, Ordering::Relaxed);
        let storage = DType::Primitive(PType::I64, initial);
        let reconstructed = ExtDType::<CountingValidator>::try_from_storage(42, storage.clone())?;
        assert_eq!(reconstructed.metadata(), &42);
        assert_eq!(reconstructed.storage_dtype(), &storage);
        assert_eq!(VALIDATIONS.load(Ordering::Relaxed), 1);
        assert_eq!(DECLARATIONS.load(Ordering::Relaxed), 1);
        let restored = CountingValidator.deserialize(&42u32.to_le_bytes(), storage)?;
        assert_eq!(restored, reconstructed.erased());
        assert_eq!(VALIDATIONS.load(Ordering::Relaxed), 2);
        assert_eq!(DECLARATIONS.load(Ordering::Relaxed), 2);
    }

    Ok(())
}

/// An extension type that declares the storage dtype that its vtable holds. A read keeps that
/// declaration whatever the stored storage dtype is.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct DeclaredStorage(DType);

impl ExtVTable for DeclaredStorage {
    type Metadata = DType;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.declared_storage");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        Ok(self.0.clone())
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        metadata.clone()
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}

/// A [`DeclaredStorage`] dtype over [`DType::Null`] storage, which is always nullable.
fn null_extension() -> DType {
    let ext_dtype = ExtDType::try_with_vtable(DeclaredStorage(DType::Null), DType::Null, Nullable)
        .vortex_expect("DeclaredStorage accepts every storage dtype");
    DType::Extension(ext_dtype.erased())
}

/// A read of a declared storage dtype without a non-nullable form, from a non-nullable stored
/// storage dtype, returns the storage mismatch error instead of panicking.
#[rstest]
#[case::null(DType::Null, "test.declared_storage storage must be null, got i32")]
#[case::extension_over_null(
    null_extension(),
    "test.declared_storage storage must be test.declared_storage[null](null), got i32"
)]
fn test_read_of_storage_without_non_nullable_form(#[case] declared: DType, #[case] expected: &str) {
    let stored = DType::Primitive(PType::I32, NonNullable);
    let result = ExtDType::try_from_storage_with_vtable(
        DeclaredStorage(declared.clone()),
        declared.clone(),
        stored.clone(),
    );
    assert!(result.is_err_and(|error| error.to_string().contains(expected)));
    let result = DeclaredStorage(declared).deserialize(&[], stored);
    assert!(result.is_err_and(|error| error.to_string().contains(expected)));
}

/// [`ExtDType::try_with_vtable`] validates the metadata before it asks for the storage
/// declaration, so a time unit that [`Date`] or [`Time`] does not support returns an error
/// instead of a panic.
#[rstest]
#[case::date_seconds(
    Date::try_new(TimeUnit::Seconds, NonNullable).map(ExtDType::erased),
    "Date time unit must be days or milliseconds, got s"
)]
#[case::date_microseconds(
    Date::try_new(TimeUnit::Microseconds, NonNullable).map(ExtDType::erased),
    "Date time unit must be days or milliseconds, got µs"
)]
#[case::date_nanoseconds(
    Date::try_new(TimeUnit::Nanoseconds, NonNullable).map(ExtDType::erased),
    "Date time unit must be days or milliseconds, got ns"
)]
#[case::time_days(
    Time::try_new(TimeUnit::Days, NonNullable).map(ExtDType::erased),
    "Time time unit must be finer than days, got days"
)]
fn test_unsupported_time_unit_returns_error(
    #[case] result: VortexResult<ExtDTypeRef>,
    #[case] expected: &str,
) {
    assert!(result.is_err_and(|error| error.to_string().contains(expected)));
}

/// Foreign metadata normalizes outer nullability and preserves nested nullability, so equality
/// and hashing that ignore nullability still distinguish different nested storage declarations.
#[rstest]
#[case::top_level(NonNullable, Nullable, true)]
#[case::nested(Nullable, NonNullable, false)]
fn test_foreign_dtype_ignores_only_top_level_nullability(
    #[case] element: Nullability,
    #[case] list: Nullability,
    #[case] expected: bool,
) -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.foreign");
    let i32_list = |element, list| DType::list(DType::Primitive(PType::I32, element), list);
    let dtype = ForeignExtDType::from_parts(*ID, vec![], i32_list(NonNullable, NonNullable))?;
    let other = ForeignExtDType::from_parts(*ID, vec![], i32_list(element, list))?;
    let (dtype, other) = (DType::Extension(dtype), DType::Extension(other));

    assert_eq!(dtype.eq_ignore_nullability(&other), expected);

    // Dtypes that are equal ignoring nullability must also hash equally ignoring nullability.
    if expected {
        let hash = |dtype: &DType| {
            let mut hasher = DefaultHasher::new();
            dtype.hash_ignore_nullability(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(&dtype), hash(&other));
    }

    Ok(())
}

#[rstest]
#[case::compact(false)]
#[case::pretty(true)]
fn test_debug_preserves_metadata(#[case] pretty: bool) -> VortexResult<()> {
    let dtype =
        Timestamp::try_new_with_tz(TimeUnit::Milliseconds, Some("UTC".into()), Nullable)?.erased();
    let output = if pretty {
        format!("{dtype:#?}")
    } else {
        format!("{dtype:?}")
    };

    assert!(output.starts_with("ExtDType {"), "{output}");
    assert!(output.contains("metadata: TimestampOptions {"), "{output}");
    assert!(output.contains("unit: Milliseconds"), "{output}");
    assert_eq!(
        output.contains("metadata: TimestampOptions {\n"),
        pretty,
        "{output}"
    );
    assert_eq!(output.matches("UTC").count(), 1, "{output}");

    Ok(())
}

#[test]
fn storage_construction_validates_before_declaration() {
    let storage = DType::Primitive(PType::I32, NonNullable);
    let error = ExtDType::<Date>::try_from_storage(TimeUnit::Seconds, storage)
        .expect_err("Date rejects seconds before requesting its storage declaration");
    assert!(
        error
            .to_string()
            .contains("Date time unit must be days or milliseconds")
    );
}

/// Null storage supports nullable construction and rejects non-nullable requests without panicking.
#[rstest]
#[case::null(DType::Null)]
#[case::extension_over_null(null_extension())]
fn metadata_first_null_storage(#[case] declared: DType) -> VortexResult<()> {
    let vtable = DeclaredStorage(declared.clone());
    let dtype = ExtDType::try_with_vtable(vtable.clone(), declared.clone(), Nullable)?;
    assert_eq!(dtype.storage_dtype(), &declared);

    let expected =
        format!("test.declared_storage storage must have a non-nullable form, got {declared}");
    let error = ExtDType::try_with_vtable(vtable.clone(), declared.clone(), NonNullable)
        .expect_err("null storage has no non-nullable form");
    assert!(error.to_string().contains(&expected), "{error}");

    let storage = ConstantArray::new(Scalar::null(declared.clone()), 1).into_array();
    ExtensionArray::try_new_from_vtable(vtable.clone(), declared.clone(), storage)?;

    let storage = ConstantArray::new(1i32, 1).into_array();
    let error = ExtensionArray::try_new_from_vtable(vtable, declared, storage)
        .expect_err("null storage cannot match non-nullable i32 storage");
    assert!(error.to_string().contains(&expected), "{error}");

    Ok(())
}

#[test]
fn null_elements_allow_non_nullable_list_storage() -> VortexResult<()> {
    let declared = DType::list(DType::Null, Nullable);
    let dtype =
        ExtDType::try_with_vtable(DeclaredStorage(declared.clone()), declared, NonNullable)?;

    assert_eq!(
        dtype.storage_dtype(),
        &DType::list(DType::Null, NonNullable)
    );

    Ok(())
}
