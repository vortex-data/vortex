use std::any::Any;
use std::sync::Arc;

use vortex_array::aliases::hash_map::HashMap;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};

use crate::IDENTITY_IDENTIFIER;

pub type Identifier = Arc<str>;
type ExprScope<T> = HashMap<Identifier, T>;

#[derive(Clone, Default)]
pub struct Scope {
    array_len: usize,
    root_scope: Option<ArrayRef>,
    /// A map from identifiers to arrays
    values: ExprScope<ArrayRef>,
    #[allow(dead_code)]
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

    pub fn values(&self, id: &Identifier) -> VortexResult<&ArrayRef> {
        if id.as_ref() == IDENTITY_IDENTIFIER {
            return self
                .root_scope
                .as_ref()
                .ok_or_else(|| vortex_err!("no root scope"));
        }
        self.values
            .get(id)
            .ok_or_else(|| vortex_err!("cannot find {} in values scope", id))
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
        self.clone().with_value(ident, value)
    }

    pub fn with_value(mut self, ident: impl Into<Identifier>, value: ArrayRef) -> Self {
        assert_eq!(value.len(), self.len());
        let ident = ident.into();

        if ident.as_ref() == IDENTITY_IDENTIFIER {
            self.root_scope = Some(value);
        } else {
            self.values.insert(ident, value);
        }
        self
    }

    pub fn with_var(mut self, ident: Identifier, var: Arc<dyn Any + Send + Sync>) -> Self {
        self.vars.insert(ident, var);
        self
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
                ctx.values
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

    pub fn dtype(&self, id: &Identifier) -> VortexResult<&DType> {
        if id.as_ref() == IDENTITY_IDENTIFIER {
            return self
                .root
                .as_ref()
                .ok_or_else(|| vortex_err!("missing root type"));
        }
        self.types
            .get(id)
            .ok_or_else(|| vortex_err!("cannot find {} in values scope", id))
    }

    pub fn copy_with_value(&self, ident: Identifier, dtype: DType) -> Self {
        self.clone().with_value(ident, dtype)
    }

    pub fn with_value(mut self, ident: Identifier, dtype: DType) -> Self {
        if ident.as_ref() == IDENTITY_IDENTIFIER {
            self.root = Some(dtype);
        } else {
            self.types.insert(ident, dtype);
        }
        self
    }
}
