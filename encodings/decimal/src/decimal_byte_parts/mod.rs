mod serde;

use vortex_array::arrays::DecimalArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::{DecimalArrayTrait, PrimitiveArrayTrait};
use vortex_array::vtable::{ComputeVTable, VTableRef};
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayOperationsImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, ArrayVariants, ArrayVariantsImpl, Canonical, Encoding, ProstMetadata,
    try_from_array_ref,
};
use vortex_dtype::{DType, DecimalDType, match_each_signed_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::{DecimalValue, Scalar};

use crate::decimal_wrapper::serde::DecimalWrapperMetadata;

/// This array shorts decimals as between 1-4 columns of primitive children.
/// All but the most significant column will be unsigned ints, whereas the final one
/// is signed.
///
/// An i64 decimal value will be sorted as single i64 column
/// An i128 decimal value will be sorted as two array columns with ptype: [i64, u64].
#[derive(Clone, Debug)]
pub struct DecimalBytePartsArray {
    encoded: ArrayRef,
    dtype: DType,
    stats_set: ArrayStats,
}

impl DecimalBytePartsArray {
    pub fn try_new(array: ArrayRef, decimal_dtype: DecimalDType) -> VortexResult<Self> {
        // For now only signed integer types are supported, this can be relaxed in the future.
        if !array.dtype().is_signed_int() {
            vortex_bail!("decimal wrapper can only wrap integer dtypes")
        }

        let primitive = array
            .as_primitive_typed()
            .vortex_expect("checked is primitive");

        if decimal_dtype.bit_width() > primitive.ptype().bit_width() {
            vortex_bail!(
                "cannot fit a decimal {decimal_dtype} into a primitive with ptype {}",
                primitive.ptype()
            )
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
    type Array = DecimalBytePartsArray;
    type Metadata = ProstMetadata<DecimalWrapperMetadata>;
}

impl ArrayImpl for DecimalBytePartsArray {
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

        DecimalBytePartsArray::try_new(child.clone(), *self.decimal_dtype())
    }
}

impl DecimalArrayTrait for DecimalBytePartsArray {}

impl ArrayVariantsImpl for DecimalBytePartsArray {
    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        Some(self)
    }
}

impl ArrayCanonicalImpl for DecimalBytePartsArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        let prim = self.encoded.to_canonical()?.into_primitive()?;
        // Depending on the decimal type and the min/max of the primitive array we can choose
        // the correct buffer size

        match_each_signed_integer_ptype!(prim.ptype(), |$P| {
           Ok(Canonical::Decimal(DecimalArray::new(
                prim.buffer::<$P>(),
                self.decimal_dtype().clone(),
                prim.validity().clone(),
            )))
        })
    }
}

impl ArrayOperationsImpl for DecimalBytePartsArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DecimalBytePartsArray::try_new(self.encoded.slice(start, stop)?, *self.decimal_dtype())
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

impl ArrayStatisticsImpl for DecimalBytePartsArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for DecimalBytePartsArray {
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
