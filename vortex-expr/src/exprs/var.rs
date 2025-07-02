use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::stats::Stat;
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata};
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_proto::exprs as pb;

use crate::{
    AccessPath, AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, Identifier, Scope, ScopeDType,
    StatsCatalog, VTable, VortexExpr, vtable,
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
        ExprEncodingRef::new_ref(&VarExprEncoding)
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        let var = match &expr.identifier {
            Identifier::Identity => "".to_string(),
            Identifier::Other(var) => var.to_string(),
        };
        Some(ProstMetadata(pb::VarOpts { var }))
    }

    fn children(_expr: &Self::Expr) -> Vec<ExprRef> {
        vec![]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        if !children.is_empty() {
            return vortex_bail!("Var expression does not have children, got: {:?}", children);
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

        let var = if metadata.var.is_empty() {
            Identifier::Identity
        } else {
            Identifier::from(metadata.var.clone())
        };

        Ok(VarExpr { identifier: var })
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
    pub fn new_expr(identifier: Identifier) -> ExprRef {
        Arc::new(Self { identifier })
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
    VarExpr::new_expr(ident.into())
}

/// Return a global pointer to the identity token.
/// This is the name of the data found in a vortex array or file.
pub fn root() -> ExprRef {
    VarExpr::new_expr(Identifier::Identity)
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
