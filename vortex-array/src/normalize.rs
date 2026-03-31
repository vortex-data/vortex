// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::Id;

use crate::ArrayRef;
use crate::session::ArrayRegistry;

/// Options for normalizing an array.
pub struct NormalizeOptions<'a> {
    /// The set of allowed array encodings (in addition to the canonical ones) that are permitted
    /// in the normalized array.
    pub allowed: &'a ArrayRegistry,
    /// The operation to perform when a non-allowed encoding is encountered.
    pub operation: Operation,
}

/// The operation to perform when a non-allowed encoding is encountered.
pub enum Operation {
    Error,
    // TODO(joe): add into canonical variant
}

impl ArrayRef {
    /// Normalize the array according to given options.
    ///
    /// This operation performs a recursive traversal of the array. Any non-allowed encoding is
    /// normalized per the configured operation.
    pub fn normalize(self, options: &mut NormalizeOptions) -> VortexResult<ArrayRef> {
        let array_ids = options.allowed.ids().collect_vec();
        self.normalize_with_error(&array_ids)?;
        // Note this takes ownership so we can at a later date remove non-allowed encodings.
        Ok(self)
    }

    fn normalize_with_error(&self, allowed: &[Id]) -> VortexResult<()> {
        if !allowed.contains(&self.encoding_id()) {
            vortex_bail!(AssertionFailed: "normalize forbids encoding ({})", self.encoding_id())
        }

        for child in self.children() {
            child.normalize_with_error(allowed)?
        }
        Ok(())
    }
}
