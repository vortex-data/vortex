// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// How to measure the length of a string or binary value.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LenMode {
    /// Number of bytes in the value (SQL `octet_length`). For UTF-8 values this can be computed
    /// directly from the variable-binary view buffer without reading the data bytes themselves.
    #[default]
    Bytes,
    /// Number of UTF-8 characters in the value (SQL `character_length`/`length`). This requires
    /// inspecting the value bytes to count codepoints.
    Chars,
}

impl Display for LenMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LenMode::Bytes => write!(f, "bytes"),
            LenMode::Chars => write!(f, "chars"),
        }
    }
}

/// Expression that computes the length of each string or binary value, returning a `u64` column.
///
/// The key optimization this enables is projection pushdown: a query that only needs
/// `length(str)` can compute the lengths during the scan and materialize a narrow integer column
/// instead of the full (potentially very wide) string column.
#[derive(Clone)]
pub struct Len;

impl ScalarFnVTable for Len {
    type Options = LenMode;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.len")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let tag = match instance {
            LenMode::Bytes => 0u8,
            LenMode::Chars => 1u8,
        };
        Ok(Some(vec![tag]))
    }

    fn deserialize(&self, metadata: &[u8], _session: &VortexSession) -> VortexResult<Self::Options> {
        match metadata.first() {
            Some(0) | None => Ok(LenMode::Bytes),
            Some(1) => Ok(LenMode::Chars),
            Some(other) => vortex_bail!("Invalid LenMode tag {other}"),
        }
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Len expression", child_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        if !input.is_utf8() && !input.is_binary() {
            vortex_bail!("len expression requires Utf8 or Binary input dtype, got {input}");
        }
        Ok(DType::Primitive(PType::U64, input.nullability()))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?;
        let validity = child.validity()?;
        let canonical = child.execute::<Canonical>(ctx)?;
        let varbinview = canonical.as_varbinview();
        let views = varbinview.views();

        let lengths: Buffer<u64> = match options {
            LenMode::Bytes => views.iter().map(|view| u64::from(view.len())).collect(),
            LenMode::Chars => (0..views.len())
                .map(|idx| {
                    let bytes = varbinview.bytes_at(idx);
                    // Count UTF-8 codepoints: every byte that is not a continuation byte
                    // (0b10xx_xxxx) starts a new character.
                    bytes
                        .as_slice()
                        .iter()
                        .filter(|&&b| (b & 0xC0) != 0x80)
                        .count() as u64
                })
                .collect(),
        };

        Ok(PrimitiveArray::new(lengths, validity).into_array())
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _instance: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::IntoArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::octet_len;
    use crate::expr::char_len;
    use crate::expr::root;
    use crate::validity::Validity;

    #[test]
    fn byte_len() {
        let arr = VarBinViewArray::from_iter_str(["a", "bb", "", "this is a long string"]).into_array();
        let result = arr.apply(&octet_len(root())).unwrap();
        assert_eq!(result.dtype(), &DType::Primitive(PType::U64, Nullability::NonNullable));
        assert_arrays_eq!(
            result,
            PrimitiveArray::new(vortex_buffer::buffer![1u64, 2, 0, 21], Validity::NonNullable)
        );
    }

    #[test]
    fn char_len_multibyte() {
        // "é" is two bytes but one character; "你好" is six bytes but two characters.
        let arr = VarBinViewArray::from_iter_str(["é", "你好", "abc"]).into_array();
        let result = arr.apply(&char_len(root())).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::new(vortex_buffer::buffer![1u64, 2, 3], Validity::NonNullable)
        );
    }

    #[test]
    fn preserves_nulls() {
        let arr =
            VarBinViewArray::from_iter_nullable_str([Some("abc"), None, Some("de")]).into_array();
        let result = arr.apply(&octet_len(root())).unwrap();
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::U64, Nullability::Nullable)
        );
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(3u64), None, Some(2)])
        );
    }
}
