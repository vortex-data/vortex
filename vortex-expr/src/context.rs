use std::any::Any;
use std::iter;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err};

use crate::IDENTITY_IDENTIFIER;

pub type Identifier = Arc<str>;
pub type ExprScope<T> = HashMap<Identifier, T>;

#[allow(dead_code)]
#[derive(Clone, Default)]
pub struct ValuesScope(ExprScope<ArrayRef>);
#[allow(dead_code)]
#[derive(Clone, Default)]
pub struct ValuesDTypeScope(ExprScope<DType>);
#[allow(dead_code)]
#[derive(Clone, Default)]
pub struct VarsScope(ExprScope<Arc<dyn Any>>);

#[derive(Clone, Default)]
pub struct EvaluationContext {
    array_len: usize,
    #[allow(dead_code)]
    /// A map from identifiers to arrays
    values: ValuesScope,
    #[allow(dead_code)]
    /// A map identifiers to opaque values used by expressions, but
    /// cannot affect the result type/shape.
    vars: VarsScope,
}

impl EvaluationContext {
    pub fn new(values: ValuesScope, vars: VarsScope) -> VortexResult<Self> {
        // This could default to len = 0?
        let len = values
            .0
            .values()
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("cannot have any empty evaluation context"))?
            .len();
        assert!(values.0.values().map(|a| a.len()).all_equal());
        Ok(Self {
            array_len: len,
            values,
            vars,
        })
    }

    pub fn default_scope(arr: ArrayRef) -> Self {
        Self::try_from(ValuesScope::default_array(arr)).vortex_expect("cannot fail")
    }

    pub fn values(&self, id: &Identifier) -> VortexResult<&ArrayRef> {
        self.values
            .0
            .get(id)
            .ok_or_else(|| vortex_err!("cannot find {} in values scope", id))
    }

    pub fn len(&self) -> usize {
        self.array_len
    }

    pub fn with(&self, ident: Identifier, value: ArrayRef) -> Self {
        assert_eq!(value.len(), self.len());

        let values = ValuesScope(HashMap::from_iter(
            self.values
                .0
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .chain(iter::once((ident, value))),
        ));

        Self {
            array_len: self.array_len,
            values,
            vars: self.vars.clone(),
        }
    }
}

impl TryFrom<ValuesScope> for EvaluationContext {
    type Error = VortexError;

    fn try_from(values: ValuesScope) -> VortexResult<Self> {
        Self::new(values, VarsScope::default())
    }
}

impl ValuesScope {
    pub fn default_array(arr: ArrayRef) -> Self {
        Self(HashMap::from([(Arc::from("$"), arr)]))
    }

    pub fn new(scope: ExprScope<ArrayRef>) -> Self {
        Self(scope)
    }
}

#[derive(Clone, Default)]
pub struct DTypeEvaluationContext {
    types: ValuesDTypeScope,
}

impl From<&EvaluationContext> for DTypeEvaluationContext {
    fn from(ctx: &EvaluationContext) -> Self {
        Self {
            types: ValuesDTypeScope(HashMap::from_iter(
                ctx.values
                    .0
                    .iter()
                    .map(|(k, v)| (k.clone(), v.dtype().clone())),
            )),
        }
    }
}

impl DTypeEvaluationContext {
    pub fn new_identity(dtype: DType) -> Self {
        Self {
            types: ValuesDTypeScope(HashMap::from([(IDENTITY_IDENTIFIER.into(), dtype)])),
        }
    }

    pub fn type_(&self, id: &Identifier) -> VortexResult<&DType> {
        self.types
            .0
            .get(id)
            .ok_or_else(|| vortex_err!("cannot find {} in values scope", id))
    }

    pub fn with(&self, ident: Identifier, type_: DType) -> Self {
        let values = ValuesDTypeScope(HashMap::from_iter(
            self.types
                .0
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .chain(iter::once((ident, type_))),
        ));

        Self { types: values }
    }
}
