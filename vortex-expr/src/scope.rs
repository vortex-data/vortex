use std::any::Any;
use std::str::FromStr;
use std::sync::Arc;

use vortex_array::aliases::hash_map::HashMap;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};

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

impl Identifier {
    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Identifier::Identity => write!(f, ""),
            Identifier::Other(v) => write!(f, "{}", v),
        }
    }
}

/// The evaluation scope for an expression, all variables are evaluated relatively to the data it contains.
/// It allows for expressions to access fields and previously defined data by [`Identifier`].
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

    pub fn copy_with_value(&self, ident: Identifier, value: ArrayRef) -> Self {
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

#[derive(Clone, Default)]
pub struct ScopeDType {
    root: Option<DType>,
    types: ExprScope<DType>,
}

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

    pub fn copy_with_value(&self, ident: Identifier, dtype: DType) -> Self {
        self.clone().with_value(ident, dtype)
    }

    pub fn with_value(mut self, ident: Identifier, dtype: DType) -> Self {
        if ident.is_identity() {
            self.root = Some(dtype);
        } else {
            self.types.insert(ident, dtype);
        }
        self
    }
}
