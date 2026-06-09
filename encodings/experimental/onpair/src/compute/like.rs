// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `LIKE` pushdown for OnPair, evaluating `prefix%` and `%needle%` patterns
//! directly on the compressed code stream via a per-code DFA (see [`crate::dfa`]).

use std::sync::LazyLock;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
#[cfg(test)]
use std::sync::atomic::Ordering;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::decode::collect_widened;
use crate::dfa::OnPairMatcher;

/// Test-only counter of how many times the kernel actually committed to the
/// compressed-domain pushdown (i.e. returned `Some` via the DFA path). Lets
/// tests assert that the pushdown genuinely fires through the execution engine,
/// including through wrappers like `Dict` and `Shared`.
#[cfg(test)]
pub(crate) static PUSHDOWN_HITS: AtomicUsize = AtomicUsize::new(0);

/// Escape hatch for measuring or debugging the compressed-domain LIKE pushdown:
/// set `VORTEX_ONPAIR_LIKE_PUSHDOWN=0` (or `off`/`false`) to force the kernel to
/// decline, falling back to canonical decompression + LIKE. Read once.
static PUSHDOWN_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    matches!(
        std::env::var("VORTEX_ONPAIR_LIKE_PUSHDOWN").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
});

impl LikeKernel for OnPair {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if *PUSHDOWN_DISABLED {
            return Ok(None);
        }

        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };

        if options.case_insensitive {
            return Ok(None);
        }

        let pattern_bytes: &[u8] = if let Some(s) = pattern_scalar.as_utf8_opt() {
            let Some(v) = s.value() else {
                return Ok(None);
            };
            v.as_ref()
        } else if let Some(b) = pattern_scalar.as_binary_opt() {
            let Some(v) = b.value() else {
                return Ok(None);
            };
            v
        } else {
            return Ok(None);
        };

        let dict_bytes = array.dict_bytes();
        let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;

        let Some(matcher) = OnPairMatcher::try_new(
            dict_bytes.as_slice(),
            dict_offsets.as_slice(),
            pattern_bytes,
        )?
        else {
            return Ok(None);
        };

        #[cfg(test)]
        PUSHDOWN_HITS.fetch_add(1, Ordering::Relaxed);

        let negated = options.negated;
        let n = array.len();

        // `codes_offsets` are per-row boundaries into the (possibly sliced)
        // `codes` child. A sliced OnPair keeps the full `codes` child and only
        // narrows these offsets, so `offsets[0]` may be > 0; slice the `codes`
        // window to `[offsets[0], offsets[n])` before materialising it, exactly
        // as the canonical decoder does.
        let offsets = array
            .codes_offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let (code_start, code_end): (usize, usize) =
            match_each_integer_ptype!(offsets.ptype(), |T| {
                let s = offsets.as_slice::<T>();
                (
                    AsPrimitive::<usize>::as_(s[0]),
                    AsPrimitive::<usize>::as_(s[n]),
                )
            });

        let codes = collect_widened::<u16>(&array.codes().slice(code_start..code_end)?, ctx)?;
        let codes = codes.as_slice();

        let result = match_each_integer_ptype!(offsets.ptype(), |T| {
            let off = offsets.as_slice::<T>();
            matcher.scan_to_bitbuf(n, off, code_start, codes, negated)
        });

        let validity = array
            .array()
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
}
