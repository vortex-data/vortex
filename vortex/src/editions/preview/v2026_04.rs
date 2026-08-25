// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The April 2026 `preview` component cohort.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The April 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_04_0: EditionId = EditionId::new("preview", 2026, 4, 0);

/// The declaration of [`PREVIEW_2026_04_0`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_04_0,
        min_vortex_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.patched"),
        EditionMember::array(&"vortex.tensor.cosine_similarity"),
        EditionMember::array(&"vortex.tensor.inner_product"),
        EditionMember::array(&"vortex.tensor.normalized"),
        EditionMember::array(&"vortex.tensor.l2_norm"),
        EditionMember::dtype(&"vortex.tensor.fixed_shape_tensor"),
        EditionMember::dtype(&"vortex.tensor.vector"),
    ],
};
