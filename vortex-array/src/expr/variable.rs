// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use crate::dtype::FieldName;
use crate::expr::Expression;

/// The name of a value bound in a [`Scope`](crate::expr::Scope).
#[derive(Clone, Debug)]
pub struct Variable(Arc<str>);

impl Variable {
    /// Create a variable with the given name.
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Arc::from(name.as_ref()))
    }

    /// The variable's name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// Compares by name, with a pointer-equality fast path for the common case of a cloned handle.
impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for Variable {}

impl Hash for Variable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the name, not the pointer, so equal names hash equally.
        self.0.hash(state);
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Variable> for Expression {
    fn from(variable: Variable) -> Self {
        Expression::Variable(variable)
    }
}

impl From<FieldName> for Variable {
    fn from(field_name: FieldName) -> Self {
        Variable(field_name.take())
    }
}

impl From<&str> for Variable {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl AsRef<str> for Variable {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_is_by_name_not_identity() {
        let x1 = Variable::new("x");
        let x2 = Variable::new("x");
        let y = Variable::new("y");

        // Independently constructed, so no shared allocation to short-circuit on.
        assert_eq!(x1, x2);
        assert_ne!(x1, y);
    }

    #[test]
    fn equal_variables_hash_equally() {
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;

        let state = RandomState::new();
        assert_eq!(
            state.hash_one(Variable::new("x")),
            state.hash_one(Variable::new("x"))
        );
    }

    #[test]
    fn cloning_shares_the_name() {
        let original = Variable::new("x");
        let cloned = original.clone();
        assert!(Arc::ptr_eq(&original.0, &cloned.0));
    }
}
