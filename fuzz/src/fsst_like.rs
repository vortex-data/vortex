// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer for FSST LIKE pushdown: compresses arbitrary strings with FSST, then
//! runs a LIKE pattern on both the compressed and uncompressed arrays, asserting
//! that the boolean results are identical.

use std::sync::LazyLock;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_error::VortexResult;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// A random string from a small alphabet (`a..=h`) with bounded length.
#[derive(Debug)]
struct SmallAlphabetString {
    max_len: usize,
}

impl SmallAlphabetString {
    fn generate(&self, u: &mut Unstructured<'_>) -> arbitrary::Result<String> {
        let len: usize = u.int_in_range(0..=self.max_len)?;
        (0..len)
            .map(|_| Ok(u.int_in_range(b'a'..=b'h')? as char))
            .collect()
    }
}

/// Fuzz input: a set of strings and a LIKE pattern.
#[derive(Debug)]
pub struct FuzzFsstLike {
    pub strings: Vec<String>,
    pub pattern: String,
    pub negated: bool,
}

impl<'a> Arbitrary<'a> for FuzzFsstLike {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let n_strings: usize = u.int_in_range(1..=200)?;
        let str_gen = SmallAlphabetString { max_len: 512 };
        let strings: Vec<String> = (0..n_strings)
            .map(|_| str_gen.generate(u))
            .collect::<arbitrary::Result<_>>()?;

        let needle = SmallAlphabetString { max_len: 254 }.generate(u)?;

        let pattern = match u.int_in_range(0..=2)? {
            0 => format!("{needle}%"),  // prefix
            1 => format!("%{needle}%"), // contains
            2 => format!("%{needle}"),  // suffix (should fall back, still correct)
            _ => unreachable!(""),
        };

        let negated: bool = u.arbitrary()?;

        Ok(FuzzFsstLike {
            strings,
            pattern,
            negated,
        })
    }
}

/// Run the FSST LIKE fuzzer: compare LIKE on compressed vs uncompressed.
///
/// Returns:
/// - `Ok(true)` — keep in corpus
/// - `Ok(false)` — reject (e.g. too few strings)
/// - `Err(_)` — mismatch found (bug)
#[expect(clippy::result_large_err)]
pub fn run_fsst_like_fuzz(fuzz: FuzzFsstLike) -> VortexFuzzResult<bool> {
    let FuzzFsstLike {
        strings,
        pattern,
        negated,
    } = fuzz;

    if strings.is_empty() {
        return Ok(false);
    }

    let len = strings.len();

    // Build uncompressed VarBinArray.
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );

    // Train FSST compressor and compress.
    let compressor = fsst_train_compressor(&varbin);
    let mut ctx = SESSION.create_execution_ctx();
    let fsst_array: FSSTArray = fsst_compress(
        varbin.clone(),
        varbin.len(),
        varbin.dtype(),
        &compressor,
        &mut ctx,
    );

    let opts = LikeOptions {
        negated,
        case_insensitive: false,
    };

    // Run LIKE on the uncompressed array.
    let expected = run_like_on_array(&varbin.into_array(), &pattern, len, opts)
        .map_err(|err| VortexFuzzError::VortexError(err, Backtrace::capture()))?;

    // Run LIKE on the FSST-compressed array.
    let actual = run_like_on_array(&fsst_array.into_array(), &pattern, len, opts)
        .map_err(|err| VortexFuzzError::VortexError(err, Backtrace::capture()))?;

    // Compare bit-for-bit.
    let expected_bits = expected.to_bit_buffer();
    let actual_bits = actual.to_bit_buffer();
    for idx in 0..len {
        let expected_val = expected_bits.value(idx);
        let actual_val = actual_bits.value(idx);
        if expected_val != actual_val {
            return Err(VortexFuzzError::VortexError(
                vortex_error::vortex_err!(
                    "FSST LIKE mismatch at index {idx}:\n  \
                     pattern:  {pattern:?}\n  \
                     string:   {:?}\n  \
                     expected: {expected_val}\n  \
                     actual:   {actual_val}",
                    &strings[idx],
                ),
                Backtrace::capture(),
            ));
        }
    }

    Ok(true)
}

fn run_like_on_array(
    array: &ArrayRef,
    pattern: &str,
    len: usize,
    opts: LikeOptions,
) -> VortexResult<BoolArray> {
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;

    let pattern_arr = ConstantArray::new(pattern, len).into_array();
    let result = Like
        .try_new_array(len, opts, [array.clone(), pattern_arr])?
        .into_array()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
    Ok(result.into_bool())
}
