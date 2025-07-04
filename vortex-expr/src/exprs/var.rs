use std::fmt::Display;

use vortex_array::stats::Stat;
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata};
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;

use crate::{
    AccessPath, AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, Identifier, IntoExpr, Scope,
    ScopeDType, StatsCatalog, VTable, vtable,
};

vtable!(Var);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VarExpr {
    identifier: Identifier,
}

pub struct VarExprEncoding;

impl VTable for VarVTable {
    type Expr = VarExpr;
    type Encoding = VarExprEncoding;
    type Metadata = ProstMetadata<pb::VarOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("var")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(VarExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        let var = match &expr.identifier {
            Identifier::Identity => "".to_string(),
            Identifier::Other(var) => var.to_string(),
        };
        Some(ProstMetadata(pb::VarOpts { var }))
    }

    fn children(_expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        if !children.is_empty() {
            vortex_bail!("Var expression does not have children, got: {:?}", children);
        }
        Ok(expr.clone())
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if !children.is_empty() {
            vortex_bail!("Var expression does not have children, got: {:?}", children);
        }
        Ok(VarExpr {
            identifier: Identifier::from(metadata.var.as_str()),
        })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        scope
            .array(&expr.identifier)
            .cloned()
            .ok_or_else(|| vortex_err!("cannot find '{}' in arrays scope", expr.identifier))
    }

    fn return_dtype(expr: &Self::Expr, scope: &ScopeDType) -> VortexResult<DType> {
        scope
            .dtype(&expr.identifier)
            .cloned()
            .ok_or_else(|| vortex_err!("cannot find '{}' in dtype scope", expr.identifier))
    }
}

/// Used to extract values (Arrays from the Scope).
/// see `Scope`.
impl VarExpr {
    pub fn new(identifier: Identifier) -> Self {
        Self { identifier }
    }

    pub fn var(&self) -> &Identifier {
        &self.identifier
    }
}

impl Display for VarExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.identifier)
    }
}

impl AnalysisExpr for VarExpr {
    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Max)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Min)
    }

    fn field_path(&self) -> Option<AccessPath> {
        Some(AccessPath::new(FieldPath::root(), self.identifier.clone()))
    }
}

pub fn var(ident: impl Into<Identifier>) -> ExprRef {
    VarExpr::new(ident.into()).into_expr()
}

/// Return a global pointer to the identity token.
/// This is the name of the data found in a vortex array or file.
pub fn root() -> ExprRef {
    VarExpr::new(Identifier::Identity).into_expr()
}

pub fn is_root(expr: &ExprRef) -> bool {
    expr.as_opt::<VarVTable>()
        .is_some_and(|v| v.var().is_identity())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::{Identifier, Scope, eq, var};

    #[test]
    fn test_two_vars() {
        let a1 = PrimitiveArray::new(buffer![5, 4, 3, 2, 1, 0], Validity::AllValid).to_array();
        let a2 = PrimitiveArray::from_iter(1..=6).to_array();

        let expr = eq(var(Identifier::Identity), var("row"));
        let res = expr
            .evaluate(&Scope::new(a1).with_array("row".parse().unwrap(), a2))
            .unwrap();
        let res = res.to_bool().unwrap().boolean_buffer().iter().collect_vec();

        assert_eq!(res, vec![false, false, true, false, false, false])
    }

    #[test]
    fn test_empty_string_ident_not_allowed() {
        assert!(Identifier::from_str("").is_err());
    }
}
