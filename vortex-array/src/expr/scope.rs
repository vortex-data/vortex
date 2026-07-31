// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::DType;

/// The context an [`Expression`](crate::expr::Expression) is bound against.
///
/// Today a scope is just the dtype that [`root`](crate::expr::root) resolves to. It is an opaque
/// struct rather than a bare [`DType`] so that lexical bindings can be added later without changing
/// [`Expression::bind_scope`](crate::expr::Expression::bind_scope)'s signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    root: DType,
}

impl Scope {
    /// Create a scope in which `root` resolves to the given dtype.
    pub fn new(root: DType) -> Self {
        Self { root }
    }

    /// The dtype that `root` resolves to.
    pub fn root(&self) -> &DType {
        &self.root
    }
}

impl From<DType> for Scope {
    fn from(root: DType) -> Self {
        Self::new(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Nullability;

    #[test]
    fn root_round_trips() {
        let dtype = DType::Bool(Nullability::Nullable);
        assert_eq!(Scope::new(dtype.clone()).root(), &dtype);
        assert_eq!(Scope::from(dtype.clone()).root(), &dtype);
    }
}
