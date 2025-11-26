// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// A trait for describing the signature of a scalar function.
pub trait Signature {
    /// Returns the arity (number of arguments) for this function.
    fn arity(&self) -> usize;

    /// Returns the display name of the nth child/argument for this function.
    fn child_name(&self, child_idx: usize) -> Option<String>;
}

/// A simply unary signature implementation.
pub struct UnarySignature;
impl Signature for UnarySignature {
    fn arity(&self) -> usize {
        1
    }

    fn child_name(&self, _child_idx: usize) -> Option<String> {
        Some("input".to_string())
    }
}
