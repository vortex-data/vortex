use std::fmt::Display;

use serde::{Deserialize, Serialize};
use vortex_error::{vortex_panic, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::aliases::hash_map::HashMap;
use crate::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use crate::encoding::ids;
use crate::stats::{Stat, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::{impl_encoding, ArrayDType, ArrayTrait};

mod canonical;
mod compute;
mod stats;
mod variants;

impl_encoding!("vortex.constant", ids::CONSTANT, Constant);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantMetadata {
    scalar_value: ScalarValue,
}

impl Display for ConstantMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConstantMetadata {{ scalar_value: {} }}",
            self.scalar_value
        )
    }
}

impl ConstantArray {
    pub fn new<S>(scalar: S, length: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        // TODO(aduffy): add stats for bools, ideally there should be a
        //  StatsSet::constant(Scalar) constructor that does this for us, like StatsSet::nulls.
        let stats = StatsSet::from(HashMap::from([
            (Stat::Max, scalar.clone()),
            (Stat::Min, scalar.clone()),
            (Stat::IsConstant, true.into()),
            (Stat::IsSorted, true.into()),
            (Stat::RunCount, 1.into()),
        ]));

        Self::try_from_parts(
            scalar.dtype().clone(),
            length,
            ConstantMetadata {
                scalar_value: scalar.value().clone(),
            },
            [].into(),
            stats,
        )
        .unwrap_or_else(|err| {
            vortex_panic!(
                err,
                "Failed to create Constant array of length {} from scalar {}",
                length,
                scalar
            )
        })
    }

    pub fn scalar_value(&self) -> &ScalarValue {
        &self.metadata().scalar_value
    }

    /// Construct an owned [`vortex_scalar::Scalar`] with a value equal to [`Self::scalar_value()`].
    pub fn owned_scalar(&self) -> Scalar {
        Scalar::new(self.dtype().clone(), self.scalar_value().clone())
    }
}

impl ArrayTrait for ConstantArray {}

impl ArrayValidity for ConstantArray {
    fn is_valid(&self, _index: usize) -> bool {
        !self.scalar_value().is_null()
    }

    fn logical_validity(&self) -> LogicalValidity {
        match self.scalar_value().is_null() {
            true => LogicalValidity::AllInvalid(self.len()),
            false => LogicalValidity::AllValid(self.len()),
        }
    }
}

impl AcceptArrayVisitor for ConstantArray {
    fn accept(&self, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        Ok(())
    }
}
