// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The edition list and per-encoding membership declarations.
//!
//! Published editions are frozen: the computed set of a published edition is pinned by a
//! golden test in this crate, so any change to these declarations that alters a published
//! set fails CI. New encodings are staged into the current draft edition.

use crate::Edition;
use crate::EditionId;
use crate::EncodingDecl;

/// The first edition of the `core` family: the encodings the default file writer emits.
pub const CORE_2026_07_0: EditionId = EditionId::new("core", 2026, 7, 0);

/// The draft successor of [`CORE_2026_07_0`]. Carries no guarantee until frozen.
pub const CORE_2026_10_0: EditionId = EditionId::new("core", 2026, 10, 0);

/// All editions, oldest first within each family. The newest non-draft edition of each
/// family is that family's `current` edition.
pub static EDITIONS: &[Edition] = &[
    Edition {
        id: CORE_2026_07_0,
        draft: false,
    },
    Edition {
        id: CORE_2026_10_0,
        draft: true,
    },
];

const fn encoding(id: &'static str, since: EditionId) -> EncodingDecl {
    EncodingDecl {
        id,
        since,
        // TODO(editions): record per-encoding required releases from compat-fixture
        //  evidence; until then the edition's required release falls back to the first
        //  release containing the edition.
        required_vortex_release: None,
    }
}

/// Membership declarations for every encoding covered by an edition.
pub static ENCODINGS: &[EncodingDecl] = &[
    encoding("vortex.null", CORE_2026_07_0),
    encoding("vortex.bool", CORE_2026_07_0),
    encoding("vortex.primitive", CORE_2026_07_0),
    encoding("vortex.decimal", CORE_2026_07_0),
    encoding("vortex.varbin", CORE_2026_07_0),
    encoding("vortex.varbinview", CORE_2026_07_0),
    encoding("vortex.list", CORE_2026_07_0),
    encoding("vortex.listview", CORE_2026_07_0),
    encoding("vortex.fixed_size_list", CORE_2026_07_0),
    encoding("vortex.struct", CORE_2026_07_0),
    encoding("vortex.variant", CORE_2026_07_0),
    encoding("vortex.ext", CORE_2026_07_0),
    encoding("vortex.chunked", CORE_2026_07_0),
    encoding("vortex.constant", CORE_2026_07_0),
    encoding("vortex.dict", CORE_2026_07_0),
    encoding("vortex.masked", CORE_2026_07_0),
    encoding("vortex.sparse", CORE_2026_07_0),
    encoding("vortex.alp", CORE_2026_07_0),
    encoding("vortex.alprd", CORE_2026_07_0),
    encoding("vortex.bytebool", CORE_2026_07_0),
    encoding("vortex.datetimeparts", CORE_2026_07_0),
    encoding("vortex.decimal_byte_parts", CORE_2026_07_0),
    encoding("vortex.fsst", CORE_2026_07_0),
    encoding("vortex.pco", CORE_2026_07_0),
    encoding("vortex.runend", CORE_2026_07_0),
    encoding("vortex.sequence", CORE_2026_07_0),
    encoding("vortex.zigzag", CORE_2026_07_0),
    encoding("vortex.zstd", CORE_2026_07_0),
    encoding("fastlanes.bitpacked", CORE_2026_07_0),
    encoding("fastlanes.delta", CORE_2026_07_0),
    encoding("fastlanes.for", CORE_2026_07_0),
    encoding("fastlanes.rle", CORE_2026_07_0),
];
