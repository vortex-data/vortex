// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::len::LenMode;
use vortex_array::scalar_fn::fns::len::LenReduce;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;

impl LenReduce for FSST {
    fn len(array: ArrayView<'_, Self>, mode: LenMode) -> VortexResult<Option<ArrayRef>> {
        // FSST stores the uncompressed *byte* length of every value as a dedicated child, so the
        // octet length is available directly — without consulting the symbol table or expanding
        // any of the compressed codes.
        if mode != LenMode::Bytes {
            // Character length needs the value bytes to count codepoints.
            return Ok(None);
        }

        // Nulls are tracked in a separate validity child, not in the lengths child, so the stored
        // lengths only map directly onto the result for non-nullable arrays. Fall back otherwise
        // (the default `len` execution canonicalizes) to keep validity correct.
        if array.dtype().is_nullable() {
            return Ok(None);
        }

        let lengths = array
            .uncompressed_lengths()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?;
        Ok(Some(lengths))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::octet_len;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::fns::len::LenMode;
    use vortex_array::scalar_fn::fns::len::LenReduce;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FSST;
    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn make_fsst(strings: &[&str]) -> FSSTArray {
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(*s)),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        let len = varbin.len();
        let dtype = varbin.dtype().clone();
        fsst_compress(
            varbin,
            len,
            &dtype,
            &compressor,
            &mut SESSION.create_execution_ctx(),
        )
    }

    /// The reduce rule returns the stored lengths directly (no decompression).
    #[test]
    fn reduce_returns_lengths() -> VortexResult<()> {
        let fsst = make_fsst(&["a", "bb", "", "hello world"]);
        let reduced = <FSST as LenReduce>::len(fsst.as_view(), LenMode::Bytes)?;
        assert!(reduced.is_some(), "octet_len over FSST should be reduced");
        let result = reduced
            .unwrap()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_slice::<u64>(), &[1, 2, 0, 11]);
        Ok(())
    }

    /// Character length needs the bytes, so the reduce declines.
    #[test]
    fn reduce_declines_chars() -> VortexResult<()> {
        let fsst = make_fsst(&["a", "bb"]);
        let reduced = <FSST as LenReduce>::len(fsst.as_view(), LenMode::Chars)?;
        assert!(reduced.is_none(), "char_len over FSST should fall back");
        Ok(())
    }

    /// End-to-end: applying `octet_len` to an FSST array produces the right lengths.
    #[test]
    fn octet_len_end_to_end() -> VortexResult<()> {
        let fsst = make_fsst(&["a", "bb", "", "hello world"]).into_array();
        let result = fsst
            .apply(&octet_len(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_primitive();
        assert_arrays_eq!(
            result,
            PrimitiveArray::new(
                vortex_buffer::buffer![1u64, 2, 0, 11],
                vortex_array::validity::Validity::NonNullable
            )
        );
        Ok(())
    }
}
