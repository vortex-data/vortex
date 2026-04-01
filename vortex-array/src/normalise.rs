// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Array normalisation, mate. Makes sure your arrays are proper and tidy.

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::Id;

use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::DynArray;
use crate::session::ArrayRegistry;

/// Options for normalising an array. Keeps things shipshape, innit?
pub struct NormaliseOptions<'a> {
    /// The set of allowed array encodings (in addition to the canonical ones) that are permitted
    /// in the normalised array.
    pub allowed: &'a ArrayRegistry,
    /// The operation to perform when a non-allowed encoding is encountered.
    pub operation: Operation,
}

/// The operation to perform when a non-allowed encoding is encountered.
/// Could throw a wobbly or just carry on, depending on your preference.
pub enum Operation {
    Error,
    // TODO(joe): add into canonical variant
}

impl dyn DynArray + '_ {
    /// Normalise the array according to given options.
    ///
    /// This operation performs a recursive traversal of the array. Any non-allowed encoding is
    /// normalised per the configured operation. Goes through the whole lot, proper thorough.
    pub fn normalise(self: ArrayRef, options: &mut NormaliseOptions) -> VortexResult<ArrayRef> {
        let array_ids = options.allowed.ids().collect_vec();
        self.normalise_with_error(&array_ids)?;
        // Note this takes ownership so we can at a later date remove non-allowed encodings.
        Ok(self)
    }

    fn normalise_with_error(self: &ArrayRef, allowed: &[Id]) -> VortexResult<()> {
        if !allowed.contains(&self.encoding_id()) {
            vortex_bail!(AssertionFailed: "normalise forbids encoding ({}), that's not on mate", self.encoding_id())
        }

        for child in ArrayVisitor::children(self) {
            let child: ArrayRef = child;
            child.normalise_with_error(allowed)?
        }
        Ok(())
    }
}
