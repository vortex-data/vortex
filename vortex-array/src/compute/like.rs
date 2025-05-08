use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::arrow::{Datum, from_arrow_array_with_len};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

/// Perform SQL left LIKE right
///
/// There are two wildcards supported with the LIKE operator:
/// - %: matches zero or more characters
/// - _: matches exactly one character
pub fn like(
    array: &dyn Array,
    pattern: &dyn Array,
    options: LikeOptions,
) -> VortexResult<ArrayRef> {
    LIKE_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), pattern.into()],
            options: &options,
        })?
        .unwrap_array()
}

pub struct LikeKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(LikeKernelRef);

pub trait LikeKernel: Encoding {
    fn like(
        &self,
        array: &Self::Array,
        pattern: &dyn Array,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug)]
pub struct LikeKernelAdapter<E: Encoding>(pub E);

impl<E: Encoding + LikeKernel> LikeKernelAdapter<E> {
    pub const fn lift(&'static self) -> LikeKernelRef {
        LikeKernelRef(ArcRef::new_ref(self))
    }
}

impl<E: Encoding + LikeKernel> Kernel for LikeKernelAdapter<E> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = LikeArgs::try_from(args)?;
        let Some(array) = inputs.array.as_any().downcast_ref::<E::Array>() else {
            return Ok(None);
        };
        Ok(E::like(&self.0, array, inputs.pattern, inputs.options)?.map(|array| array.into()))
    }
}

pub static LIKE_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("like".into(), ArcRef::new_ref(&Like));
    for kernel in inventory::iter::<LikeKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct Like;

impl ComputeFnVTable for Like {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let LikeArgs {
            array,
            pattern,
            options,
        } = LikeArgs::try_from(args)?;

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&LIKE_FN, args)? {
            return Ok(output);
        }

        // Otherwise, we fall back to the Arrow implementation
        Ok(arrow_like(array, pattern, options)?.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let LikeArgs { array, pattern, .. } = LikeArgs::try_from(args)?;
        if !matches!(array.dtype(), DType::Utf8(..)) {
            vortex_bail!("Expected utf8 array, got {}", array.dtype());
        }
        if !matches!(pattern.dtype(), DType::Utf8(..)) {
            vortex_bail!("Expected utf8 pattern, got {}", array.dtype());
        }
        let nullability = array.dtype().is_nullable() || pattern.dtype().is_nullable();
        Ok(DType::Bool(nullability.into()))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let LikeArgs { array, pattern, .. } = LikeArgs::try_from(args)?;
        if array.len() != pattern.len() {
            vortex_bail!(
                "Length mismatch lhs len {} ({}) != rhs len {} ({})",
                array.len(),
                array.encoding(),
                pattern.len(),
                pattern.encoding()
            );
        }
        Ok(array.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

/// Options for SQL LIKE function
#[derive(Default, Debug, Clone, Copy)]
pub struct LikeOptions {
    pub negated: bool,
    pub case_insensitive: bool,
}

impl Options for LikeOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct LikeArgs<'a> {
    array: &'a dyn Array,
    pattern: &'a dyn Array,
    options: LikeOptions,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for LikeArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("Expected 2 inputs, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected first input to be an array"))?;
        let pattern = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Expected second input to be an array"))?;
        let options = *value
            .options
            .as_any()
            .downcast_ref::<LikeOptions>()
            .vortex_expect("Expected options to be LikeOptions");

        Ok(LikeArgs {
            array,
            pattern,
            options,
        })
    }
}

/// Implementation of `LikeFn` using the Arrow crate.
pub(crate) fn arrow_like(
    array: &dyn Array,
    pattern: &dyn Array,
    options: LikeOptions,
) -> VortexResult<ArrayRef> {
    let nullable = array.dtype().is_nullable() | pattern.dtype().is_nullable();
    let len = array.len();
    assert_eq!(
        array.len(),
        pattern.len(),
        "Arrow Like: length mismatch for {}",
        array.encoding()
    );
    let lhs = Datum::try_new(array)?;
    let rhs = Datum::try_new(pattern)?;

    let result = match (options.negated, options.case_insensitive) {
        (false, false) => arrow_string::like::like(&lhs, &rhs)?,
        (true, false) => arrow_string::like::nlike(&lhs, &rhs)?,
        (false, true) => arrow_string::like::ilike(&lhs, &rhs)?,
        (true, true) => arrow_string::like::nilike(&lhs, &rhs)?,
    };

    from_arrow_array_with_len(&result, len, nullable)
}
