// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `tensor` edition family.
//!
//! Tensor support is opt-in. This module declares its extension dtypes and persisted array
//! encodings together.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionFamily;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The `tensor` family: tensor extension dtypes and persisted tensor array encodings.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "tensor",
    origin: "vortex-tensor",
    doc: "Tensor extension dtypes and persisted tensor array encodings. Tensor support is \
opt-in: a reader built without `vortex-tensor` cannot resolve these members, so they are \
versioned independently of `core` and enabled only when the crate is initialized.",
};

/// The April 2026 draft edition of the `tensor` family.
pub const TENSOR_2026_04: EditionId = EditionId::new("tensor", 2026, 4, 0);

/// The declaration of [`TENSOR_2026_04`] and the tensor components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: TENSOR_2026_04,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.tensor.cosine_similarity"),
        EditionMember::array(&"vortex.tensor.inner_product"),
        EditionMember::array(&"vortex.tensor.l2_norm"),
        EditionMember::array(&"vortex.tensor.normalized"),
        EditionMember::dtype(&"vortex.tensor.fixed_shape_tensor"),
        EditionMember::dtype(&"vortex.tensor.vector"),
    ],
};

#[cfg(test)]
mod tests {
    use vortex_edition::EditionError;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::test_harness::validate_edition;

    use super::*;

    #[test]
    fn tensor_edition_is_valid() -> Result<(), EditionError> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        validate_edition(&session.editions(), &TENSOR_2026_04)
    }
}
