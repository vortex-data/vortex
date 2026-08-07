// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

/// The name of a value bound in a [`Scope`](crate::expr::Scope).
///
/// Deliberately distinct from [`FieldName`](crate::dtype::FieldName): a variable and a struct field
/// live in different namespaces, and giving them the same type would let one be passed where the
/// other is meant.
///
/// The same type is used in both positions a variable appears — as a parameter of
/// [`Expression::Lambda`](crate::expr::Expression::Lambda), which introduces the name, and inside
/// [`Expression::Variable`](crate::expr::Expression::Variable), which references it.
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

impl<T: AsRef<str>> From<T> for Variable {
    fn from(name: T) -> Self {
        Self::new(name)
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
