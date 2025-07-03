// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::stats::Stat;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexResult, vortex_err};

use crate::{
    AccessPath, AnalysisExpr, ExprRef, Identifier, Scope, ScopeDType, StatsCatalog, VortexExpr,
};

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Var {
    var: Identifier,
}

/// Used to extract values (Arrays from the Scope).
/// see `Scope`.
impl Var {
    pub fn new_expr(var: Identifier) -> ExprRef {
        Arc::new(Self { var })
    }

    pub fn var(&self) -> &Identifier {
        &self.var
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind::{Kind, Var as ProtoVar};

    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id, Var, root};

    // NOTE(aduffy): identity expression is deprecated for the moment, but it is still
    // in the protobuf definition. We map it into the new Var(root()) expression.
    pub(crate) struct IdentitySerde;

    impl Id for IdentitySerde {
        fn id(&self) -> &'static str {
            "identity"
        }
    }

    impl ExprDeserialize for IdentitySerde {
        fn deserialize(&self, kind: &Kind, _children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::Identity(..) = kind else {
                vortex_bail!("wrong kind {:?}, wanted identity", kind)
            };

            Ok(root())
        }
    }

    pub(crate) struct VarSerde;

    impl Id for VarSerde {
        fn id(&self) -> &'static str {
            "var"
        }
    }

    impl ExprDeserialize for VarSerde {
        fn deserialize(&self, kind: &Kind, _children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::Var(op) = kind else {
                vortex_bail!("wrong kind {:?}, wanted var", kind)
            };

            match op.var.as_str() {
                "" => Ok(Var::new_expr(crate::Identifier::Identity)),
                other => Ok(Var::new_expr(other.parse()?)),
            }
        }
    }

    impl ExprSerializable for Var {
        fn id(&self) -> &'static str {
            VarSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::Var(ProtoVar {
                var: self.var.to_string(),
            }))
        }
    }
}

impl Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.var)
    }
}

impl AnalysisExpr for Var {
    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Max)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Min)
    }

    fn field_path(&self) -> Option<AccessPath> {
        Some(AccessPath::new(FieldPath::root(), self.var.clone()))
    }
}

impl VortexExpr for Var {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, ctx: &Scope) -> VortexResult<ArrayRef> {
        ctx.array(&self.var)
            .cloned()
            .ok_or_else(|| vortex_err!("cannot find '{}' in arrays scope", self.var))
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        Var::new_expr(self.var.clone())
    }

    fn return_dtype(&self, dt_ctx: &ScopeDType) -> VortexResult<DType> {
        dt_ctx
            .dtype(&self.var)
            .cloned()
            .ok_or_else(|| vortex_err!("cannot find '{}' in dtype scope", self.var))
    }
}

pub fn var(ident: impl Into<Identifier>) -> ExprRef {
    Var::new_expr(ident.into())
}

/// Return a global pointer to the identity token.
/// This is the name of the data found in a vortex array or file.
pub fn root() -> ExprRef {
    Var::new_expr(Identifier::Identity)
}

pub fn is_root(expr: &ExprRef) -> bool {
    expr.as_any()
        .downcast_ref::<Var>()
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
