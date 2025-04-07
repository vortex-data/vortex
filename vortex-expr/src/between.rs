use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use vortex_array::compute::{BetweenOptions, between};
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::VortexResult;

use crate::{BinaryExpr, ExprRef, VortexExpr};

#[derive(Debug, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Between {
    arr: ExprRef,
    lower: ExprRef,
    upper: ExprRef,
    options: BetweenOptions,
}

impl Between {
    pub fn between(
        arr: ExprRef,
        lower: ExprRef,
        upper: ExprRef,
        options: BetweenOptions,
    ) -> ExprRef {
        Arc::new(Self {
            arr,
            lower,
            upper,
            options,
        })
    }

    pub fn to_binary_expr(&self) -> ExprRef {
        let lhs = BinaryExpr::new_expr(
            self.lower.clone(),
            self.options.lower_strict.to_operator().into(),
            self.arr.clone(),
        );
        let rhs = BinaryExpr::new_expr(
            self.arr.clone(),
            self.options.upper_strict.to_operator().into(),
            self.upper.clone(),
        );
        BinaryExpr::new_expr(lhs, crate::Operator::And, rhs)
    }
}

impl Display for Between {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({} {} {} {} {})",
            self.lower,
            self.options.lower_strict.to_operator(),
            self.arr,
            self.options.upper_strict.to_operator(),
            self.upper
        )
    }
}

impl PartialEq for Between {
    fn eq(&self, other: &Between) -> bool {
        self.arr.eq(&other.arr)
            && other.lower.eq(&self.lower)
            && other.upper.eq(&self.upper)
            && self.options == other.options
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_array::compute::{BetweenOptions, StrictComparison};
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

    use crate::between::Between;
    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct BetweenSerde;

    impl Id for BetweenSerde {
        fn id(&self) -> &'static str {
            "between"
        }
    }

    impl ExprSerializable for Between {
        fn id(&self) -> &'static str {
            BetweenSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::Between(kind::Between {
                lower_strict: self.options.lower_strict == StrictComparison::Strict,
                upper_strict: self.options.upper_strict == StrictComparison::Strict,
            }))
        }
    }

    impl ExprDeserialize for BetweenSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::Between(between) = kind else {
                vortex_bail!("wrong kind {:?}, want between", kind)
            };

            Ok(Between::between(
                children[0].clone(),
                children[1].clone(),
                children[2].clone(),
                BetweenOptions {
                    lower_strict: if between.lower_strict {
                        StrictComparison::Strict
                    } else {
                        StrictComparison::NonStrict
                    },
                    upper_strict: if between.upper_strict {
                        StrictComparison::Strict
                    } else {
                        StrictComparison::NonStrict
                    },
                },
            ))
        }
    }
}

impl VortexExpr for Between {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        let arr_val = self.arr.evaluate(batch)?;
        let lower_arr_val = self.lower.evaluate(batch)?;
        let upper_arr_val = self.upper.evaluate(batch)?;

        between(&arr_val, &lower_arr_val, &upper_arr_val, &self.options)
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.arr, &self.lower, &self.upper]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        Arc::new(Self {
            arr: children[0].clone(),
            lower: children[1].clone(),
            upper: children[2].clone(),
            options: self.options.clone(),
        })
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let arr_dt = self.arr.return_dtype(scope_dtype)?;
        let lower_dt = self.lower.return_dtype(scope_dtype)?;
        let upper_dt = self.upper.return_dtype(scope_dtype)?;

        assert!(arr_dt.eq_ignore_nullability(&lower_dt));
        assert!(arr_dt.eq_ignore_nullability(&upper_dt));

        Ok(Bool(
            arr_dt.nullability() | lower_dt.nullability() | upper_dt.nullability(),
        ))
    }
}
