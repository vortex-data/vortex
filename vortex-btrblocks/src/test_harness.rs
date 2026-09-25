// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sessions for tests and benchmarks that compress with every scheme.
//!
//! [`BtrBlocksCompressorBuilder::from_session`](crate::BtrBlocksCompressorBuilder::from_session)
//! only keeps schemes whose encodings are registered on the session and permitted by its enabled
//! editions, so a bare [`array_session`](vortex_array::array_session) compresses nothing.

use vortex_array::ArrayId;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::session::ArraySessionExt;
use vortex_edition::ComponentKind;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

const TEST_EDITION: EditionId = EditionId::new("test", 2026, 7, 0);

/// An [`array_session`](vortex_array::array_session) with [`register_encodings`] and
/// [`enable_all_registered_encodings`] applied.
pub fn session() -> VortexSession {
    let session = vortex_array::array_session();
    register_encodings(&session);
    enable_all_registered_encodings(&session);
    session
}

/// Registers every encoding the compressor's schemes can produce, with their kernels.
///
/// This mirrors `vortex_file::register_default_encodings`, copied rather than called so that
/// `vortex-btrblocks` does not depend on `vortex-file` (which depends on this crate). Keep the
/// two in step when encodings are added. `bytebool` and `tensor` are the only entries omitted:
/// this crate does not depend on them, so the compressor cannot emit them.
pub fn register_encodings(session: &VortexSession) {
    vortex_fsst::initialize(session);
    vortex_onpair::initialize(session);
    vortex_zigzag::initialize(session);
    #[cfg(feature = "zstd")]
    vortex_zstd::initialize(session);

    {
        let arrays = session.arrays();
        #[cfg(feature = "pco")]
        arrays.register(vortex_pco::Pco);
        if use_experimental_patches() {
            arrays.register(Patched);
        }
    }

    vortex_alp::initialize(session);
    vortex_datetime_parts::initialize(session);
    vortex_decimal_byte_parts::initialize(session);
    vortex_fastlanes::initialize(session);
    vortex_runend::initialize(session);
    vortex_sequence::initialize(session);
    vortex_sparse::initialize(session);
}

/// Declares and enables a test edition containing every array encoding registered on `session`.
pub fn enable_all_registered_encodings(session: &VortexSession) {
    let ids = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    enable_encodings(session, ids);
}

/// Declares and enables a test edition containing exactly the array encodings `ids`.
pub fn enable_encodings(session: &VortexSession, ids: impl IntoIterator<Item = ArrayId>) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: TEST_EDITION,
            min_library_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is valid");
    for id in ids {
        editions
            .declare_inclusion(EditionInclusion::new(
                ComponentKind::Array,
                &id,
                TEST_EDITION,
            ))
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("array has one test-edition inclusion");
    }
    session
        .enable_edition(TEST_EDITION)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is registered");
}
