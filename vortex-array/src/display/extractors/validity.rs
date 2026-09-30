// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Structural validity annotations for array trees.
//!
//! Only validity that is already known from the representation is displayed. Array-backed validity
//! is left unannotated so rendering never computes an aggregate.

use std::fmt;

use crate::ArrayRef;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::validity::Validity;

/// Extractor that adds structural validity to the header without scanning array values.
pub struct ValidityExtractor;

impl TreeExtractor<ArrayRef, TreeContext> for ValidityExtractor {
    fn write_header(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if !array.dtype().is_nullable() {
            return Ok(());
        }

        match array.validity() {
            Ok(Validity::NonNullable | Validity::AllValid) => f.write_str(" [all_valid]"),
            Ok(Validity::AllInvalid) => f.write_str(" [all_invalid]"),
            Ok(Validity::Array(_)) => Ok(()),
            Err(e) => {
                tracing::warn!("Failed to check validity: {e}");
                f.write_str(" [validity_failed]")
            }
        }
    }
}
