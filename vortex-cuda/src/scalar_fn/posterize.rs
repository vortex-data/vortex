// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Posterize scalar function that quantizes uint8 values to discrete levels.
//!
//! For example, `posterize(col("R"), 4)` maps each byte to one of {0, 85, 170, 255}.

use std::fmt;
use std::fmt::Formatter;

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::PrimitiveVTable;
use vortex::array::arrays::ScalarFnArrayExt;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::Expression;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::session::VortexSession;

/// Posterize options specifying the number of discrete levels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PosterizeOptions {
    /// Number of quantization levels (e.g. 4 → {0, 85, 170, 255}).
    pub levels: u32,
}

impl fmt::Display for PosterizeOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "levels={}", self.levels)
    }
}

/// Scalar function that quantizes uint8 values to a fixed number of evenly spaced levels.
#[derive(Clone)]
pub struct Posterize;

impl ScalarFnVTable for Posterize {
    type Options = PosterizeOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.posterize")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Posterize expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "posterize(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", {})", options.levels)
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let child_dtype = &arg_dtypes[0];
        if !matches!(child_dtype, DType::Primitive(PType::U8, _)) {
            vortex_bail!(
                "Posterize expects a u8 primitive child, got: {}",
                child_dtype
            );
        }
        Ok(child_dtype.clone())
    }

    fn execute(&self, options: &Self::Options, mut args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let child = args.inputs.pop().vortex_expect("Missing input child");
        let levels = options.levels;

        if let Some(prim) = child.as_opt::<PrimitiveVTable>() {
            let input = prim.as_slice::<u8>();
            #[allow(clippy::cast_possible_truncation)]
            let output: Buffer<u8> = input
                .iter()
                .map(|&v| {
                    let v = u32::from(v);
                    let mut bucket = v * levels / 256;
                    if bucket >= levels {
                        bucket = levels - 1;
                    }
                    // bucket is in [0, levels-1], so result is always in [0, 255]
                    (bucket * 255 / (levels - 1)) as u8
                })
                .collect();
            return Ok(PrimitiveArray::new(output, prim.validity()?).into_array());
        }

        // Execute child to canonical form and wrap in a new ScalarFnArray
        let executed = child.execute::<ArrayRef>(args.ctx)?;
        let len = executed.len();
        Posterize.try_new_array(len, options.clone(), [executed])
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.levels.to_le_bytes().to_vec()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let levels = u32::from_le_bytes(
            metadata
                .try_into()
                .map_err(|_| vortex::error::vortex_err!("Invalid posterize metadata"))?,
        );
        Ok(PosterizeOptions { levels })
    }
}

/// Creates a posterize expression that quantizes uint8 values to the given number of levels.
pub fn posterize(child: Expression, levels: u32) -> Expression {
    Posterize.new_expr(PosterizeOptions { levels }, [child])
}
