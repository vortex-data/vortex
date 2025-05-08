mod serde;

use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::DecimalArrayTrait;
use vortex_array::vtable::{ComputeVTable, VTableRef};
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayOperationsImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, ArrayVariantsImpl, Canonical, Encoding, ProstMetadata, try_from_array_ref,
};
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::{DecimalValue, Scalar};

use crate::decimal_wrapper::serde::DecimalWrapperMetadata;

#[derive(Clone, Debug)]
pub struct DecimalWrapperArray {
    encoded: ArrayRef,
    dtype: DType,
    stats_set: ArrayStats,
}

impl DecimalWrapperArray {
    pub fn try_new(array: ArrayRef, decimal_dtype: DecimalDType) -> VortexResult<Self> {
        if !array.dtype().is_int() {
            vortex_bail!("decimal wrapper can only wrap integer dtypes")
        }

        let nullable = array.dtype().nullability();
        Ok(Self {
            encoded: array,
            dtype: DType::Decimal(decimal_dtype, nullable),
            stats_set: Default::default(),
        })
    }

    pub fn decimal_dtype(&self) -> &DecimalDType {
        self.dtype
            .as_decimal()
            .vortex_expect("must be a decimal dtype")
    }
}

#[derive(Debug)]
pub struct DecimalWrapperEncoding;

impl ComputeVTable for DecimalWrapperEncoding {}

impl Encoding for DecimalWrapperEncoding {
    type Array = DecimalWrapperArray;
    type Metadata = ProstMetadata<DecimalWrapperMetadata>;
}

impl ArrayImpl for DecimalWrapperArray {
    type Encoding = DecimalWrapperEncoding;

    fn _len(&self) -> usize {
        self.encoded.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DecimalWrapperEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let [child] = children else {
            vortex_bail!("must replace only the single child")
        };

        DecimalWrapperArray::try_new(child.clone(), *self.decimal_dtype())
    }
}

impl DecimalArrayTrait for DecimalWrapperArray {}

impl ArrayVariantsImpl for DecimalWrapperArray {
    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        Some(self)
    }
}

impl ArrayCanonicalImpl for DecimalWrapperArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        todo!()
    }
}

impl ArrayOperationsImpl for DecimalWrapperArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DecimalWrapperArray::try_new(self.encoded.slice(start, stop)?, *self.decimal_dtype())
            .map(|d| d.to_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let scalar = self.encoded.scalar_at(index)?;

        let res = scalar
            .as_primitive()
            .typed_value::<i32>()
            .map(|v| Scalar::new(self.dtype().clone(), DecimalValue::from(v).into()))
            .unwrap_or_else(|| Scalar::null(self.dtype().clone()));
        Ok(res)
    }
}

impl ArrayStatisticsImpl for DecimalWrapperArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for DecimalWrapperArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.encoded.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.encoded.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.encoded.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.encoded.validity_mask()
    }
}

try_from_array_ref!(DecimalWrapperArray);
