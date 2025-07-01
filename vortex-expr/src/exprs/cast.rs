use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::compute::cast as compute_cast;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_proto::exprs as pb;

use crate::{
    AnalysisExpr, ExprEncoding, ExprEncodingRef, ExprId, ExprRef, Scope, ScopeDType, VortexExpr,
};

#[derive(Debug, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Cast {
    target: DType,
    child: ExprRef,
}

pub struct CastEncoding;

impl ExprEncoding for CastEncoding {
    fn id(&self) -> ExprId {
        ExprId::new_ref("cast")
    }

    fn deserialize(&self, options: &[u8], children: Vec<ExprRef>) -> VortexResult<Option<ExprRef>> {
        let options = pb::CastOpts::decode(options)?;
        let target: DType = options
            .target
            .as_ref()
            .ok_or_else(|| vortex_err!("empty target dtype"))?
            .try_into()?;
        Ok(Some(Cast::new_expr(children[0].clone(), target)))
    }
}

impl Cast {
    pub fn new_expr(child: ExprRef, target: DType) -> ExprRef {
        Arc::new(Self { target, child })
    }
}

impl PartialEq for Cast {
    fn eq(&self, other: &Self) -> bool {
        self.target.eq(&other.target) && self.child.eq(&other.child)
    }
}

impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cast({}, {})", self.child, self.target)
    }
}

impl AnalysisExpr for Cast {}

impl VortexExpr for Cast {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encoding(&self) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(&CastEncoding)
    }

    fn serialize_options(&self) -> Option<Vec<u8>> {
        Some(
            pb::CastOpts {
                target: Some((&self.target).into()),
            }
            .encode_to_vec(),
        )
    }

    fn unchecked_evaluate(&self, scope: &Scope) -> VortexResult<ArrayRef> {
        let array = self.child.evaluate(scope)?;
        compute_cast(&array, &self.target)
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.child]
    }

    fn replacing_children(self: Arc<Self>, mut children: Vec<ExprRef>) -> ExprRef {
        Self::new_expr(
            children
                .pop()
                .vortex_expect("Cast::replacing_children should have one child"),
            self.target.clone(),
        )
    }

    fn return_dtype(&self, _scope_dtype: &ScopeDType) -> VortexResult<DType> {
        Ok(self.target.clone())
    }
}

pub fn cast(child: ExprRef, target: DType) -> ExprRef {
    Cast::new_expr(child, target)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{ExprRef, Scope, ScopeDType, cast, get_item, root, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            cast(root(), DType::Bool(Nullability::NonNullable))
                .return_dtype(&ScopeDType::new(dtype))
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = cast(root(), DType::Bool(Nullability::Nullable));
        let _ = expr.replacing_children(vec![root()]);
    }

    #[test]
    fn evaluate() {
        let test_array = StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array();

        let expr: ExprRef = cast(
            get_item("a", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        let result = expr.evaluate(&Scope::new(test_array)).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
    }
}
