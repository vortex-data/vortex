//! Engine-local `TimestampSum` aggregate fn.
//!
//! Hack to avoid extending vortex's `Sum` with extension-type
//! support. Accepts a `Timestamp[unit]` extension input dtype and
//! returns an i64-nullable partial in the timestamp's storage units
//! (µs, ns, etc.). Partial saturates to null on overflow, same as
//! `Sum` over i64.
//!
//! Canonical fallback: canonicalise the extension to its storage
//! primitive, then sum as i64 with checked_add. Encoding-aware
//! kernels (see [`crate::kernels::DateTimePartsTimestampSumKernel`])
//! beat this by operating on the encoded form directly.

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::Extension;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// `TimestampSum` aggregate function — Sum that accepts Timestamp
/// (or Time/Date) extension inputs.
#[derive(Clone, Debug)]
pub struct TimestampSum;

impl TimestampSum {
    pub const ID: &'static str = "engine.timestamp_sum";
}

#[derive(Debug)]
pub struct TimestampSumPartial {
    /// `Some(value)` accumulator in i64 units; `None` after overflow.
    acc: Option<i64>,
}

impl AggregateFnVTable for TimestampSum {
    type Options = EmptyOptions;
    type Partial = TimestampSumPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new(Self::ID)
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Vec::new()))
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        // Only accept extension types whose storage is integer.
        match input_dtype {
            DType::Extension(ext) => {
                let storage = ext.storage_dtype();
                if storage.is_int() {
                    Some(DType::Primitive(PType::I64, Nullability::Nullable))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(TimestampSumPartial { acc: Some(0) })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            partial.acc = None;
            return Ok(());
        }
        let Some(acc) = partial.acc else { return Ok(()) };
        let val = other.as_primitive().typed_value::<i64>().unwrap_or(0);
        partial.acc = acc.checked_add(val);
        Ok(())
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(mut acc) = partial.acc else { return Ok(()) };
        match batch {
            Columnar::Constant(c) => {
                // Constant timestamp × len → product.
                let len = c.len() as i64;
                let v = extract_timestamp_i64(c.scalar())
                    .ok_or_else(|| vortex_err!("timestamp_sum: non-i64 timestamp constant"))?;
                let prod = v.checked_mul(len);
                let Some(prod) = prod else {
                    partial.acc = None;
                    return Ok(());
                };
                partial.acc = acc.checked_add(prod);
                Ok(())
            }
            Columnar::Canonical(c) => {
                // Canonical form of an Extension timestamp is an
                // ExtensionArray wrapping a primitive. Unwrap and sum
                // its storage as i64.
                let array = c.clone().into_array();
                let Some(ext_view) = array.as_opt::<Extension>() else {
                    // Some other canonical shape — bail.
                    return Err(vortex_err!(
                        "timestamp_sum: canonical batch is {:?} not Extension",
                        array.encoding_id()
                    ));
                };
                let storage = ext_view.storage_array();
                let storage_i64 = storage
                    .clone()
                    .cast(DType::Primitive(PType::I64, storage.dtype().nullability()))?;
                let sum_scalar =
                    vortex_array::aggregate_fn::fns::sum::sum(&storage_i64, ctx)?;
                if sum_scalar.is_null() {
                    partial.acc = None;
                    return Ok(());
                }
                let val = sum_scalar
                    .as_primitive()
                    .typed_value::<i64>()
                    .ok_or_else(|| vortex_err!("timestamp_sum: sum scalar not i64"))?;
                acc = match acc.checked_add(val) {
                    Some(v) => v,
                    None => {
                        partial.acc = None;
                        return Ok(());
                    }
                };
                partial.acc = Some(acc);
                Ok(())
            }
        }
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(match partial.acc {
            Some(v) => Scalar::primitive(v, Nullability::Nullable),
            None => Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.acc = Some(0);
    }

    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.acc.is_none()
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

/// Extract the i64 storage value out of an extension `Scalar`.
pub(crate) fn extract_timestamp_i64(scalar: &Scalar) -> Option<i64> {
    if scalar.is_null() {
        return None;
    }
    scalar.as_extension().to_storage_scalar().as_primitive().as_::<i64>()
}
