// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated UUID values and metadata for Arrow transport checks.
//!
//! Versioned fixtures set the required version and variant bits before checking transport.
//! Canonical Arrow layout and metadata checks remain in the Arrow plugin tests.

mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;
    use uuid::Version;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_arrow::test_harness::DefaultExport;
    use vortex_arrow::test_harness::check_generated;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Uuid;
    use crate::UuidMetadata;

    fn check_uuid_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable = tc.draw(gs::booleans());
        let version = tc.draw(gs::sampled_from(&[
            None,
            Some(Version::Nil),
            Some(Version::Mac),
            Some(Version::Dce),
            Some(Version::Md5),
            Some(Version::Random),
            Some(Version::Sha1),
            Some(Version::SortMac),
            Some(Version::SortRand),
            Some(Version::Custom),
            Some(Version::Max),
        ]));
        let mut bytes = tc.draw(
            gs::vecs(gs::integers::<u8>())
                .min_size(len * 16)
                .max_size(len * 16),
        );

        if let Some(version) = version {
            set_uuid_version(&mut bytes, version);
        }

        let validity = if nullable {
            Validity::from_iter(tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len)))
        } else {
            Validity::NonNullable
        };
        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(bytes).into_array(),
            16,
            validity,
            len,
        )?
        .into_array();

        let array = ExtensionArray::try_new_from_vtable(Uuid, UuidMetadata { version }, storage)?
            .into_array();

        check_generated(tc, array, DefaultExport::Supported, &mut ctx)
    }

    /// Sets version and variant bits, including the special nil and max UUID encodings.
    fn set_uuid_version(bytes: &mut [u8], version: Version) {
        let (uuids, remainder) = bytes.as_chunks_mut::<16>();
        assert!(
            remainder.is_empty(),
            "UUID storage must contain complete UUIDs"
        );

        for uuid in uuids {
            match version {
                Version::Nil => uuid.fill(0),
                Version::Max => uuid.fill(255),
                _ => {
                    *uuid = uuid::Builder::from_bytes(*uuid)
                        .with_version(version)
                        .with_variant(uuid::Variant::RFC4122)
                        .into_uuid()
                        .into_bytes();
                }
            }
        }
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn arrow_transport(tc: TestCase) {
        check_uuid_round_trip(&tc)
            .vortex_expect("Arrow transport must preserve dtype and stored values");
    }
}
