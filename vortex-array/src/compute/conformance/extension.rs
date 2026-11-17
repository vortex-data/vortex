// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conformance checks for extension dtype implementations.
//!
//! The checks use caller-provided storage to test metadata reconstruction and value validation
//! independently of extension array construction.

use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::validate_each_value;

/// Checks that serialized metadata and the stored dtype reconstruct the original extension dtype,
/// and that the array validator agrees with the scalar-based default.
///
/// Pass storage that matches the extension dtype's declaration. Use both legal and illegal
/// values to exercise an override's acceptance and rejection paths.
#[track_caller]
pub fn test_ext_vtable_conformance<V: ExtVTable>(
    ext_dtype: &ExtDType<V>,
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) {
    let vtable = ext_dtype.vtable();
    let bytes = ext_dtype
        .serialize_metadata()
        .vortex_expect("extension metadata must serialize");
    let stored_metadata = vtable
        .deserialize_metadata(&bytes)
        .vortex_expect("serialized extension metadata must deserialize");
    let roundtrip = ExtDType::try_from_storage_with_vtable(
        vtable.clone(),
        stored_metadata,
        storage.dtype().clone(),
    )
    .vortex_expect("extension dtype must reconstruct from stored metadata and storage");
    assert_eq!(
        &roundtrip, ext_dtype,
        "extension dtype changed on round trip"
    );

    let expected = validate_each_value(ext_dtype, storage, ctx);
    let actual = roundtrip.erased().validate_storage_array(storage, ctx);
    assert_eq!(
        actual.is_ok(),
        expected.is_ok(),
        "{} array validation returned {actual:?}, scalar validation returned {expected:?}",
        ext_dtype.id(),
    );
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use super::test_ext_vtable_conformance;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtVTable;
    use crate::dtype::extension::ForeignExtDType;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::masked_invalid([true, false, true], true)]
    #[case::unmasked_invalid([true, true, false], false)]
    fn validate_array_default_checks_each_value(
        #[case] validity: [bool; 3],
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let dtype = Time::try_new(TimeUnit::Seconds, Nullable)?;
        let storage =
            PrimitiveArray::new(buffer![0i32, 86_401, 86_399], Validity::from_iter(validity))
                .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = Time::validate_array(&dtype, &storage, &mut ctx);
        assert_eq!(result.is_ok(), expected, "{result:?}");
        if let Err(error) = result {
            assert!(error.to_string().contains("Invalid time scalar"), "{error}");
        }
        test_ext_vtable_conformance(&dtype, &storage, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::valid(Some(86_399), 32, true)]
    #[case::invalid(Some(86_401), 32, false)]
    #[case::null(None, 32, true)]
    #[case::empty(Some(86_401), 0, true)]
    fn constant_time_conformance(
        #[case] value: Option<i32>,
        #[case] len: usize,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let dtype = Time::try_new(TimeUnit::Seconds, Nullable)?;
        let storage = ConstantArray::new(value, len).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = Time::validate_array(&dtype, &storage, &mut ctx);
        assert_eq!(result.is_ok(), expected, "{result:?}");
        test_ext_vtable_conformance(&dtype, &storage, &mut ctx);
        Ok(())
    }

    #[test]
    fn date_conformance() -> VortexResult<()> {
        let dtype = Date::try_new(TimeUnit::Days, NonNullable)?;
        let storage = buffer![-1i32, 0, 1].into_array();
        test_ext_vtable_conformance(&dtype, &storage, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn timestamp_conformance() -> VortexResult<()> {
        let dtype =
            Timestamp::try_new_with_tz(TimeUnit::Nanoseconds, Some("UTC".into()), NonNullable)?;

        // The slice exercises validation with non-canonical storage.
        let storage = buffer![0, i64::MIN, i64::MAX].into_array().slice(1..3)?;
        test_ext_vtable_conformance(&dtype, &storage, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn foreign_extension_conformance() -> VortexResult<()> {
        static ID: CachedId = CachedId::new("test.foreign_extension");
        let storage = PrimitiveArray::from_option_iter([Some(1u32), None]).into_array();
        let dtype = ExtDType::try_from_storage_with_vtable(
            ForeignExtDType::new(*ID),
            vec![0, 0xff],
            storage.dtype().clone(),
        )?;
        test_ext_vtable_conformance(&dtype, &storage, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
