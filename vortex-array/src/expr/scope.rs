// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::dtype::DType;

/// The context an [`Expression`](crate::expr::Expression) is bound against.
///
/// A scope is the dtype that [`root`](crate::expr::root) resolves to, and the session whose scalar
/// function registry resolves the function ids the expression's nodes hold. It is an opaque struct
/// so that lexical bindings can be added later without changing
/// [`Expression::bind_scope`](crate::expr::Expression::bind_scope)'s signature.
#[derive(Clone, Debug)]
pub struct Scope {
    root: DType,
    session: VortexSession,
}

impl Scope {
    /// Create a scope in which `root` resolves to the given dtype and scalar functions are looked
    /// up in `session`.
    pub fn new(root: DType, session: VortexSession) -> Self {
        Self { root, session }
    }

    /// The dtype that `root` resolves to.
    pub fn root(&self) -> &DType {
        &self.root
    }

    /// The session that resolves scalar function ids.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Nullability;

    #[test]
    fn root_round_trips() {
        let dtype = DType::Bool(Nullability::Nullable);
        assert_eq!(
            Scope::new(dtype.clone(), VortexSession::empty()).root(),
            &dtype
        );
    }
}
