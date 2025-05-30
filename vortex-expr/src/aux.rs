use std::any::Any;
use std::fmt::Display;
use std::sync::{Arc, LazyLock};

use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ExprRef, VortexExpr};

static AUX: LazyLock<ExprRef> = LazyLock::new(|| Arc::new(Aux));
pub static AUX_ID: &'static str = "aux";

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Aux;

impl Aux {
    pub fn new_expr() -> ExprRef {
        AUX.clone()
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::VortexResult;
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

    use crate::aux::{AUX, AUX_ID, Aux};
    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct AuxSerde;

    impl Id for AuxSerde {
        fn id(&self) -> &'static str {
            AUX_ID
        }
    }

    impl ExprDeserialize for AuxSerde {
        fn deserialize(&self, _expr: &Kind, _children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            Ok(AUX.clone())
        }
    }

    impl ExprSerializable for Aux {
        fn id(&self) -> &'static str {
            AuxSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::Identity(kind::Identity {}))
        }
    }
}

impl Display for Aux {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_$")
    }
}

impl VortexExpr for Aux {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        batch.to_struct()?.field_by_name("#").cloned()
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        self
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        Ok(scope_dtype.clone())
    }
}

pub fn aux() -> ExprRef {
    Aux::new_expr()
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;

    use crate::{EvalCtx, aux, ident};

    #[test]
    fn test_aux_and_arr() {
        let aux_array = PrimitiveArray::from_iter(0i32..10).to_array();
        let arr = PrimitiveArray::from_iter(10i32..20).to_array();

        let ctx = EvalCtx::new(arr, aux_array).unwrap();

        let value = ident().evaluate(&ctx).unwrap();
        let value = value.to_primitive().unwrap();
        assert_eq!(value.as_slice::<i32>(), (10..20).collect_vec().as_slice());

        let row_id = aux().evaluate(&ctx).unwrap();
        let row_id = row_id.to_primitive().unwrap();
        assert_eq!(row_id.as_slice::<i32>(), (0..10).collect_vec().as_slice());
    }
}
