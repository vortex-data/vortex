// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::session::ArraySessionExt;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// This is a vortex edition used for testing and shouldn't made public.
const TEST_EDITION: EditionId = EditionId::new("test", 2026, 7, 0);

pub fn enable_all_registered_array_encodings(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: TEST_EDITION,
            min_vortex_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is valid");
    let ids = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    for id in ids {
        editions
            .declare_inclusion(EditionInclusion::new(&id, TEST_EDITION))
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("registered array encoding has one test-edition inclusion");
    }
    session
        .enable_edition(TEST_EDITION)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is registered");
}
