// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression tests for extension dtype nullability changes and debug output.

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::ExtDType;
use super::ExtId;
use super::ExtVTable;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::ScalarValue;

/// The number of times [`CountingValidator::validate_dtype`] ran.
static VALIDATIONS: AtomicUsize = AtomicUsize::new(0);

/// An extension type that counts how often its dtype is validated.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
struct CountingValidator;

impl ExtVTable for CountingValidator {
    type Metadata = u32;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.counting_validator");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.to_le_bytes().to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        let bytes = metadata
            .try_into()
            .map_err(|_| vortex_err!("Expected four metadata bytes, got {}", metadata.len()))?;

        Ok(u32::from_le_bytes(bytes))
    }

    fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
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

// One test owns `VALIDATIONS`, so test threads cannot race on the counter.
#[test]
fn test_nullability_changes_do_not_validate_again() -> VortexResult<()> {
    for initial in [Nullability::NonNullable, Nullability::Nullable] {
        VALIDATIONS.store(0, Ordering::Relaxed);
        let dtype =
            ExtDType::<CountingValidator>::try_new(42, DType::Primitive(PType::I64, initial))?;
        assert_eq!(VALIDATIONS.load(Ordering::Relaxed), 1);

        for target in [Nullability::Nullable, Nullability::NonNullable] {
            let changed = dtype.with_nullability(target);
            assert_eq!(
                changed.storage_dtype(),
                &DType::Primitive(PType::I64, target)
            );
            assert_eq!(changed.metadata::<CountingValidator>(), &42);
            assert_eq!(changed.id(), dtype.id());
        }
        assert_eq!(VALIDATIONS.load(Ordering::Relaxed), 1);
    }

    Ok(())
}

#[rstest]
#[case::compact(false)]
#[case::pretty(true)]
fn test_debug_preserves_metadata(#[case] pretty: bool) {
    let dtype = Timestamp::new_with_tz(
        TimeUnit::Milliseconds,
        Some("UTC".into()),
        Nullability::Nullable,
    )
    .erased();
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
    assert!(!output.contains("Ok(())"), "{output}");
}
