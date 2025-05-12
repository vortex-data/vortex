use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantVTable, NullVTable};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::stats::{Precision, Stat, StatsProviderExt};
use crate::vtable::VTable;
use crate::{Array, ArrayExt};

/// Computes whether an array has constant values. If the array's encoding doesn't implement the
/// relevant VTable, it'll try and canonicalize in order to make a determination.
///
/// An array is constant IFF at least one of the following conditions apply:
/// 1. It has at least one element (**Note** - an empty array isn't constant).
/// 1. It's encoded as a [`crate::arrays::ConstantArray`] or [`crate::arrays::NullArray`]
/// 1. Has an exact statistic attached to it, saying its constant.
/// 1. Is all invalid.
/// 1. Is all valid AND has minimum and maximum statistics that are equal.
///
/// If the array has some null values but is not all null, it'll never be constant.
///
/// Returns `Ok(None)` if we could not determine whether the array is constant, e.g. if
/// canonicalization is disabled and the no kernel exists for the array's encoding.
pub fn is_constant(array: &dyn Array) -> VortexResult<Option<bool>> {
    let opts = IsConstantOpts::default();
    is_constant_opts(array, &opts)
}

/// Computes whether an array has constant values. Configurable by [`IsConstantOpts`].
///
/// Please see [`is_constant`] for a more detailed explanation of its behavior.
pub fn is_constant_opts(array: &dyn Array, options: &IsConstantOpts) -> VortexResult<Option<bool>> {
    Ok(IS_CONSTANT_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options,
        })?
        .unwrap_scalar()?
        .as_bool()
        .value())
}

pub static IS_CONSTANT_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("is_constant".into(), ArcRef::new_ref(&IsConstant));
    for kernel in inventory::iter::<IsConstantKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct IsConstant;

impl ComputeFnVTable for IsConstant {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let IsConstantArgs { array, options } = IsConstantArgs::try_from(args)?;

        // We try and rely on some easy to get stats
        if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(Stat::IsConstant) {
            return Ok(Scalar::from(Some(value)).into());
        }

        let value = is_constant_impl(array, options, kernels)?;

        // Only if we made a determination do we update the stats.
        if let Some(value) = value {
            array
                .statistics()
                .set(Stat::IsConstant, Precision::Exact(value.into()));
        }

        Ok(Scalar::from(value).into())
    }

    fn return_dtype(&self, _args: &InvocationArgs) -> VortexResult<DType> {
        // We always return a nullable boolean where `null` indicates we couldn't determine
        // whether the array is constant.
        Ok(DType::Bool(Nullability::Nullable))
    }

    fn return_len(&self, _args: &InvocationArgs) -> VortexResult<usize> {
        Ok(1)
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

fn is_constant_impl(
    array: &dyn Array,
    options: &IsConstantOpts,
    kernels: &[ArcRef<dyn Kernel>],
) -> VortexResult<Option<bool>> {
    match array.len() {
        // Our current semantics are that we can always get a value out of a constant array. We might want to change that in the future.
        0 => return Ok(Some(false)),
        // Array of length 1 is always constant.
        1 => return Ok(Some(true)),
        _ => {}
    }

    // Constant and null arrays are always constant
    if array.as_opt::<ConstantVTable>().is_some() || array.as_opt::<NullVTable>().is_some() {
        return Ok(Some(true));
    }

    let all_invalid = array.all_invalid()?;
    if all_invalid {
        return Ok(Some(true));
    }

    let all_valid = array.all_valid()?;

    // If we have some nulls, array can't be constant
    if !all_valid && !all_invalid {
        return Ok(Some(false));
    }

    // We already know here that the array is all valid, so we check for min/max stats.
    let min = array
        .statistics()
        .get_scalar(Stat::Min, array.dtype())
        .and_then(|p| p.as_exact());
    let max = array
        .statistics()
        .get_scalar(Stat::Max, array.dtype())
        .and_then(|p| p.as_exact());

    if let Some((min, max)) = min.zip(max) {
        if min == max {
            return Ok(Some(true));
        }
    }

    assert!(
        all_valid,
        "All values must be valid as an invariant of the VTable."
    );
    let args = InvocationArgs {
        inputs: &[array.into()],
        options,
    };
    for kernel in kernels {
        if let Some(output) = kernel.invoke(&args)? {
            return Ok(output.unwrap_scalar()?.as_bool().value());
        }
    }
    if let Some(output) = array.invoke(&IS_CONSTANT_FN, &args)? {
        return Ok(output.unwrap_scalar()?.as_bool().value());
    }

    log::debug!(
        "No is_constant implementation found for {}",
        array.encoding_id()
    );

    if options.canonicalize && !array.is_canonical() {
        let array = array.to_canonical()?;
        let is_constant = is_constant_opts(array.as_ref(), options)?;
        return Ok(is_constant);
    }

    // Otherwise, we cannot determine if the array is constant.
    Ok(None)
}

pub struct IsConstantKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(IsConstantKernelRef);

pub trait IsConstantKernel: VTable {
    /// # Preconditions
    ///
    /// * All values are valid
    /// * array.len() > 1
    ///
    /// Returns `Ok(None)` to signal we couldn't make an exact determination.
    fn is_constant(&self, array: &Self::Array, opts: &IsConstantOpts)
    -> VortexResult<Option<bool>>;
}

#[derive(Debug)]
pub struct IsConstantKernelAdapter<V: VTable>(pub V);

impl<V: VTable + IsConstantKernel> IsConstantKernelAdapter<V> {
    pub const fn lift(&'static self) -> IsConstantKernelRef {
        IsConstantKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + IsConstantKernel> Kernel for IsConstantKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let args = IsConstantArgs::try_from(args)?;
        let Some(array) = args.array.as_opt::<V>() else {
            return Ok(None);
        };
        let is_constant = V::is_constant(&self.0, array, args.options)?;
        Ok(Some(Scalar::from(is_constant).into()))
    }
}

struct IsConstantArgs<'a> {
    array: &'a dyn Array,
    options: &'a IsConstantOpts,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for IsConstantArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 1 {
            vortex_bail!("Expected 1 input, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;
        let options = value
            .options
            .as_any()
            .downcast_ref::<IsConstantOpts>()
            .ok_or_else(|| vortex_err!("Expected options to be of type IsConstantOpts"))?;
        Ok(Self { array, options })
    }
}

/// Configuration for [`is_constant_opts`] operations.
#[derive(Clone)]
pub struct IsConstantOpts {
    /// Should the operation make an effort to canonicalize the target array if its encoding doesn't implement [`IsConstantKernel`].
    pub canonicalize: bool,
}

impl Default for IsConstantOpts {
    fn default() -> Self {
        Self { canonicalize: true }
    }
}

impl Options for IsConstantOpts {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
