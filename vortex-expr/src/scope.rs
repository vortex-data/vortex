use std::any::Any;
use std::str::FromStr;
use std::sync::Arc;

use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, FieldPathSet};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_utils::aliases::hash_map::HashMap;

type ExprScope<T> = HashMap<Identifier, T>;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Identifier {
    Identity,
    Other(Arc<str>),
}

impl FromStr for Identifier {
    type Err = VortexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            vortex_bail!("Empty strings aren't allowed in identifiers")
        } else {
            Ok(Identifier::Other(s.into()))
        }
    }
}

impl PartialEq<str> for Identifier {
    fn eq(&self, other: &str) -> bool {
        match self {
            Identifier::Identity => other.is_empty(),
            Identifier::Other(str) => str.as_ref() == other,
        }
    }
}

impl From<&str> for Identifier {
    fn from(value: &str) -> Self {
        if value.is_empty() {
            Identifier::Identity
        } else {
            Identifier::Other(Arc::from(value))
        }
    }
}

impl Identifier {
    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Identifier::Identity => write!(f, ""),
            Identifier::Other(v) => write!(f, "{v}"),
        }
    }
}

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
#[derive(Clone, Default)]
pub struct Scope {
    array_len: usize,
    root_scope: Option<ArrayRef>,
    /// A map from identifiers to arrays
    arrays: ExprScope<ArrayRef>,
    /// A map identifiers to opaque values used by expressions, but
    /// cannot affect the result type/shape.
    vars: ExprScope<Arc<dyn Any + Send + Sync>>,
}

pub type ScopeElement = (Identifier, ArrayRef);

impl Scope {
    pub fn new(arr: ArrayRef) -> Self {
        Self {
            array_len: arr.len(),
            root_scope: Some(arr),
            ..Default::default()
        }
    }

    pub fn empty(len: usize) -> Self {
        Self {
            array_len: len,
            ..Default::default()
        }
    }

    /// Get a value out of the scope by its [`Identifier`]
    pub fn array(&self, id: &Identifier) -> Option<&ArrayRef> {
        if id.is_identity() {
            return self.root_scope.as_ref();
        }
        self.arrays.get(id)
    }

    pub fn vars(&self, id: Identifier) -> VortexResult<&Arc<dyn Any + Send + Sync>> {
        self.vars
            .get(&id)
            .ok_or_else(|| vortex_err!("cannot find {} in var scope", id))
    }

    pub fn is_empty(&self) -> bool {
        self.array_len == 0
    }

    pub fn len(&self) -> usize {
        self.array_len
    }

    pub fn copy_with_array(&self, ident: Identifier, value: ArrayRef) -> Self {
        self.clone().with_array(ident, value)
    }

    /// Register an array with an identifier in the scope, overriding any existing value stored in it.
    pub fn with_array(mut self, ident: Identifier, value: ArrayRef) -> Self {
        assert_eq!(value.len(), self.len());

        if ident.is_identity() {
            self.root_scope = Some(value);
        } else {
            self.arrays.insert(ident, value);
        }
        self
    }

    /// Register an array with an identifier in the scope, overriding any existing value stored in it.
    pub fn with_array_pair(self, (ident, value): ScopeElement) -> Self {
        self.with_array(ident, value)
    }

    pub fn with_var(mut self, ident: Identifier, var: Arc<dyn Any + Send + Sync>) -> Self {
        self.vars.insert(ident, var);
        self
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Identifier, &ArrayRef)> {
        let values = self.arrays.iter();

        self.root_scope
            .iter()
            .map(|s| (&Identifier::Identity, s))
            .chain(values)
    }
}

impl From<ArrayRef> for Scope {
    fn from(value: ArrayRef) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Default, Debug)]
pub struct ScopeDType {
    root: Option<DType>,
    types: ExprScope<DType>,
}

pub type ScopeDTypeElement = (Identifier, DType);

impl From<&Scope> for ScopeDType {
    fn from(ctx: &Scope) -> Self {
        Self {
            root: ctx.root_scope.as_ref().map(|s| s.dtype().clone()),
            types: HashMap::from_iter(
                ctx.arrays
                    .iter()
                    .map(|(k, v)| (k.clone(), v.dtype().clone())),
            ),
        }
    }
}

impl ScopeDType {
    pub fn new(dtype: DType) -> Self {
        Self {
            root: Some(dtype),
            ..Default::default()
        }
    }

    pub fn dtype(&self, id: &Identifier) -> Option<&DType> {
        if id.is_identity() {
            return self.root.as_ref();
        }
        self.types.get(id)
    }

    pub fn copy_with_dtype(&self, ident: Identifier, dtype: DType) -> Self {
        self.clone().with_dtype(ident, dtype)
    }

    pub fn with_dtype(mut self, ident: Identifier, dtype: DType) -> Self {
        if ident.is_identity() {
            self.root = Some(dtype);
        } else {
            self.types.insert(ident, dtype);
        }
        self
    }

    pub fn with_dtype_element(self, (ident, dtype): ScopeDTypeElement) -> Self {
        self.with_dtype(ident, dtype)
    }
}

#[derive(Default, Clone, Debug)]
pub struct ScopeFieldPathSet {
    root: Option<FieldPathSet>,
    sets: ExprScope<FieldPathSet>,
}

pub type ScopeFieldPathSetElement = (Identifier, FieldPathSet);

impl ScopeFieldPathSet {
    pub fn new(path_set: FieldPathSet) -> Self {
        Self {
            root: Some(path_set),
            ..Default::default()
        }
    }

    pub fn set(&self, id: &Identifier) -> Option<&FieldPathSet> {
        if id.is_identity() {
            return self.root.as_ref();
        }
        self.sets.get(id)
    }

    pub fn copy_with_set(&self, ident: Identifier, set: FieldPathSet) -> Self {
        self.clone().with_set(ident, set)
    }

    pub fn with_set(mut self, ident: Identifier, set: FieldPathSet) -> Self {
        if ident.is_identity() {
            self.root = Some(set);
        } else {
            self.sets.insert(ident, set);
        }
        self
    }

    pub fn with_set_element(self, (ident, set): ScopeFieldPathSetElement) -> Self {
        self.with_set(ident, set)
    }
}
