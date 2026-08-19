// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The June 2026 `preview` component cohort.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The June 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_06_0: EditionId = EditionId::new("preview", 2026, 6, 0);

/// The declaration of [`PREVIEW_2026_06_0`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_06_0,
        min_vortex_version: None,
    },
    added: &[
        EditionMember::layout(&"vortex.list"),
        // Written only by CUDA-enabled sessions, which register the layout through
        // `vortex_cuda::layout::register_cuda_layout`. A writer resolves layouts against the
        // enabled editions, so the GPU flat layout has to be a member to be written at all.
        EditionMember::layout(&"vortex.cuda_flat"),
    ],
};
