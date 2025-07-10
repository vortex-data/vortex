// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::TypeId;
use std::ops::Deref;

use vortex_array::arrays::StructArray;
use vortex_array::{ArrayRef, IntoArray};

use crate::scope_vars::{ScopeVar, ScopeVars};

/// Scope define the evaluation context/scope that an expression uses when being evaluated.
/// There is a special `Identifier` (`Identity`) which is used to bind the initial array being evaluated
///
/// Other identifier can be bound with variables either before execution or while executing (see `Let`).
/// Values can be extracted from the scope using the `Var` expression.
///
/// ```code
/// <let x = lit(1) in var(Identifier::Identity) + var(x), { Identity -> Primitive[1,2,3]> ->
/// <var(Identifier::Identity) + var(x), { Identity -> Primitive[1,2,3], x -> ConstantArray(1)> ->
/// <Primitive[1,2,3] + var(x), { Identity -> Primitive[1,2,3], x -> ConstantArray(1)> ->
/// <Primitive[1,2,3] + ConstantArray(1), { Identity -> Primitive[1,2,3], x -> ConstantArray(1)> ->
/// <Primitive[2,3,4], { Identity -> Primitive[1,2,3], x -> ConstantArray(1)>
/// ```
///
/// Other values can be bound before execution e.g.
///  `<var("x") + var("y") + var("z"), x -> ..., y -> ..., z -> ...>`
#[derive(Clone)]
pub struct Scope {
    root: ArrayRef,
    /// Variables that can be set on the scope during expression evaluation.
    scope_vars: ScopeVars,
}

impl Scope {
    /// Create a new scope with the given root array.
    pub fn new(root: ArrayRef) -> Self {
        Self {
            root,
            scope_vars: Default::default(),
        }
    }

    /// Create a new scope with the root array set an empty struct.
    pub fn empty(len: usize) -> Self {
        Self::new(StructArray::new_with_len(len).into_array())
    }

    /// Return the root array of the scope.
    pub fn root(&self) -> &ArrayRef {
        &self.root
    }

    /// Returns a new evaluation scope with the given variable applied.
    pub fn with_scope_var<V: ScopeVar>(mut self, var: V) -> Self {
        self.scope_vars.insert(TypeId::of::<V>(), Box::new(var));
        self
    }

    /// Returns the scope variable of type `V` if it exists.
    pub fn scope_var<V: ScopeVar>(&self) -> Option<&V> {
        self.scope_vars
            .get(&TypeId::of::<V>())
            .and_then(|boxed| (**boxed).as_any().downcast_ref::<V>())
    }

    /// Returns the mutable scope variable of type `V` if it exists.
    pub fn scope_var_mut<V: ScopeVar>(&mut self) -> Option<&mut V> {
        self.scope_vars
            .get_mut(&TypeId::of::<V>())
            .and_then(|boxed| (**boxed).as_any_mut().downcast_mut::<V>())
    }
}

impl Deref for Scope {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.root
    }
}

impl From<ArrayRef> for Scope {
    fn from(value: ArrayRef) -> Self {
        Self::new(value)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_scope_var() {
        use super::*;

        #[derive(Clone, PartialEq, Eq, Debug)]
        struct TestVar {
            value: i32,
        }

        let scope = Scope::empty(100);
        assert!(scope.scope_var::<TestVar>().is_none());

        let var = TestVar { value: 42 };
        let mut scope = scope.with_scope_var(var.clone());
        assert_eq!(scope.scope_var::<TestVar>(), Some(&var));

        scope.scope_var_mut::<TestVar>().unwrap().value = 43;
        assert_eq!(scope.scope_var::<TestVar>(), Some(&TestVar { value: 43 }));
    }
}
