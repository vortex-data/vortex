// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

use std::sync::Arc;

use parking_lot::Mutex;
use parking_lot::MutexGuard;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::builtins::IntDictScheme;
use crate::ctx::CompressorContext;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// The implicit root scheme ID for the compressor's own cascading (e.g. list offset compression).
///
/// This is the **only** [`SchemeId`] that is not auto-provided via [`SchemeExt`].
const ROOT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.root",
};

/// Child indices for the compressor's list/listview compression.
mod root_list_children {
    /// List/ListView offsets child.
    pub const OFFSETS: usize = 1;
    /// ListView sizes child.
    pub const SIZES: usize = 2;
}

/// The main compressor type implementing cascading adaptive compression.
///
/// This compressor applies adaptive compression [`Scheme`]s to arrays based on their data types and
/// characteristics. It recursively compresses nested structures like structs and lists, and chooses
/// optimal compression schemes for leaf types.
///
/// The compressor works by:
/// 1. Canonicalizing input arrays to a standard representation.
/// 2. Pre-filtering schemes by [`Scheme::matches`] and exclusion rules.
/// 3. Evaluating each matching scheme's compression ratio on a sample.
/// 4. Compressing with the best scheme and verifying the result is smaller.
///
/// No scheme may appear twice in a cascade chain. The compressor enforces this automatically
/// along with push/pull exclusion rules declared by each scheme.
#[derive(Debug, Clone)]
pub struct CascadingCompressor {
    /// The enabled compression schemes.
    schemes: Vec<&'static dyn Scheme>,

    /// Descendant exclusion rules for the compressor's own cascading (e.g. excluding Dict from
    /// list offsets).
    root_exclusions: Vec<DescendantExclusion>,

    /// Shared execution context for array operations during compression.
    ///
    /// This should have low contention as we only execute arrays one at a time during compression.
    ctx: Arc<Mutex<ExecutionCtx>>,
}

impl CascadingCompressor {
    /// Creates a new compressor with the given schemes.
    ///
    /// Root-level exclusion rules (e.g. excluding Dict from list offsets) are built
    /// automatically.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        // Root exclusion: exclude IntDict from list/listview offsets (monotonically
        // increasing data where dictionary encoding is wasteful).
        let root_exclusions = vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(root_list_children::OFFSETS),
        }];
        Self {
            schemes,
            root_exclusions,
            // TODO(connor): The caller should probably pass this in.
            ctx: Arc::new(Mutex::new(LEGACY_SESSION.create_execution_ctx())),
        }
    }

    /// Returns a mutable borrow of the execution context.
    pub fn execution_ctx(&self) -> MutexGuard<'_, ExecutionCtx> {
        self.ctx.lock()
    }

    /// Compresses a child array produced by a cascading scheme.
    ///
    /// If the cascade budget is exhausted, the canonical array is returned as-is. Otherwise,
    /// the child context is created by descending and recording the parent scheme + child
    /// index, and compression proceeds normally.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    pub fn compress_child(
        &self,
        child: &ArrayRef,
        parent_ctx: &CompressorContext,
        parent_id: SchemeId,
        child_index: usize,
    ) -> VortexResult<ArrayRef> {
        if parent_ctx.finished_cascading() {
            return Ok(child.clone());
        }

        let canonical = child
            .clone()
            .execute::<CanonicalValidity>(&mut self.execution_ctx())?
            .0;
        let compact = canonical.compact()?;

        let child_ctx = parent_ctx
            .clone()
            .descend_with_scheme(parent_id, child_index);
        self.compress_canonical(compact, child_ctx)
    }

    /// Compresses an array using cascading adaptive compression.
    ///
    /// First canonicalizes and compacts the array, then applies optimal compression schemes.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress(&self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        let canonical = array
            .clone()
            .execute::<CanonicalValidity>(&mut self.execution_ctx())?
            .0;

        // Compact it, removing any wasted space before we attempt to compress it.
        let compact = canonical.compact()?;

        self.compress_canonical(compact, CompressorContext::new())
    }

    /// Compresses a canonical array by dispatching to type-specific logic.
    ///
    /// # Errors
    ///
    /// Returns an error if compression of any sub-array fails.
    fn compress_canonical(
        &self,
        array: Canonical,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            Canonical::Bool(bool_array) => {
                self.choose_and_compress(Canonical::Bool(bool_array), ctx)
            }
            Canonical::Primitive(primitive) => {
                self.choose_and_compress(Canonical::Primitive(primitive), ctx)
            }
            Canonical::Decimal(decimal) => {
                self.choose_and_compress(Canonical::Decimal(decimal), ctx)
            }
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .iter_unmasked_fields()
                    .map(|field| self.compress(field))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity(),
                )?
                .into_array())
            }
            Canonical::List(list_view_array) => {
                if list_view_array.is_zero_copy_to_list() || list_view_array.elements().is_empty() {
                    let list_array = list_from_list_view(list_view_array)?;
                    self.compress_list_array(list_array, ctx)
                } else {
                    self.compress_list_view_array(list_view_array, ctx)
                }
            }
            Canonical::FixedSizeList(fsl_array) => {
                let compressed_elems = self.compress(fsl_array.elements())?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    fsl_array.validity(),
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::VarBinView(strings) => {
                if strings
                    .dtype()
                    .eq_ignore_nullability(&DType::Utf8(Nullability::NonNullable))
                {
                    self.choose_and_compress(Canonical::VarBinView(strings), ctx)
                } else {
                    // We do not compress binary arrays.
                    Ok(strings.into_array())
                }
            }
            Canonical::Extension(ext_array) => {
                let before_nbytes = ext_array.as_ref().nbytes();

                // Try scheme-based compression first.
                let result =
                    self.choose_and_compress(Canonical::Extension(ext_array.clone()), ctx)?;
                if result.nbytes() < before_nbytes {
                    return Ok(result);
                }

                // Otherwise, fall back to compressing the underlying storage array.
                let compressed_storage = self.compress(ext_array.storage_array())?;

                Ok(
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array(),
                )
            }
            Canonical::Variant(_) => {
                vortex_bail!("Variant arrays can not be compressed")
            }
        }
    }

    /// The main scheme-selection entry point for a single leaf array.
    ///
    /// Filters allowed schemes by [`matches`] and exclusion rules, merges their [`stats_options`]
    /// into a single [`GenerateStatsOptions`], then delegates to [`choose_scheme`] to pick the
    /// winner by estimated compression ratio.
    ///
    /// If a winner is found and its compressed output is actually smaller, that output is returned.
    /// Otherwise, the original array is returned unchanged.
    ///
    /// Empty and all-null arrays are short-circuited before any scheme evaluation.
    ///
    /// [`matches`]: Scheme::matches
    /// [`stats_options`]: Scheme::stats_options
    /// [`choose_scheme`]: Self::choose_scheme
    fn choose_and_compress(
        &self,
        canonical: Canonical,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        // If there are no schemes that we can compress into, then just return it uncompressed.
        if eligible_schemes.is_empty() {
            return Ok(array);
        }

        // Nothing to compress if empty or all-null.
        if array.is_empty() {
            return Ok(array);
        }

        if array.all_invalid()? {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        }

        let before_nbytes = array.nbytes();
        let merged_opts = eligible_schemes
            .iter()
            .fold(GenerateStatsOptions::default(), |acc, s| {
                acc.merge(s.stats_options())
            });

        let ctx = ctx.with_stats_options(merged_opts);

        let mut data = ArrayAndStats::new(array, merged_opts);

        if let Some(winner) = self.choose_scheme(&eligible_schemes, &mut data, ctx.clone())? {
            let compressed = winner.compress(self, &mut data, ctx)?;
            if compressed.nbytes() < before_nbytes {
                return Ok(compressed);
            }
        }

        // No scheme improved on the original.
        Ok(data.into_array())
    }

    /// Calls [`expected_compression_ratio`] on each candidate and returns the scheme with the
    /// highest ratio, or `None` if no scheme exceeds 1.0. Ties are broken by registration order
    /// (earlier in the list wins).
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    fn choose_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<Option<&'static dyn Scheme>> {
        let mut best: Option<(&'static dyn Scheme, f64)> = None;

        for &scheme in schemes {
            // Constant detection on a sample is a false positive: the sample being constant
            // does not mean the full array is constant.
            if ctx.is_sample() && scheme.detects_constant() {
                continue;
            }

            let ratio = scheme.expected_compression_ratio(self, data, ctx.clone())?;

            tracing::debug!(scheme = %scheme.id(), ratio, "evaluated compression ratio");

            if is_better_ratio(ratio, &best) {
                best = Some((scheme, ratio));

                // Schemes that return f64::MAX (like Constant) cannot be beat, so stop early.
                if ratio == f64::MAX {
                    break;
                }
            }
        }

        Ok(best.map(|(s, _)| s))
    }

    // TODO(connor): Lots of room for optimization here.
    /// Returns `true` if the candidate scheme should be excluded based on the cascade history and
    /// exclusion rules.
    fn is_excluded(&self, candidate: &dyn Scheme, ctx: &CompressorContext) -> bool {
        let id = candidate.id();
        let history = ctx.cascade_history();

        // Self-exclusion: no scheme appears twice in any chain.
        if history.iter().any(|&(sid, _)| sid == id) {
            return true;
        }

        let mut iter = history.iter().copied().peekable();

        // The root entry is always first in the history (if present). Check if the root has
        // excluded us.
        if let Some((_, child_idx)) = iter.next_if(|&(sid, _)| sid == ROOT_SCHEME_ID)
            && self
                .root_exclusions
                .iter()
                .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
        {
            return true;
        }

        // Push rules: Check if any of our ancestors have excluded us.
        for (ancestor_id, child_idx) in iter {
            if let Some(ancestor) = self.schemes.iter().find(|s| s.id() == ancestor_id)
                && ancestor
                    .descendant_exclusions()
                    .iter()
                    .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
            {
                return true;
            }
        }

        // Pull rules: Check if we have excluded ourselves because of our ancestors.
        for rule in candidate.ancestor_exclusions() {
            if history
                .iter()
                .any(|(sid, cidx)| *sid == rule.ancestor && rule.children.contains(*cidx))
            {
                return true;
            }
        }

        false
    }

    /// Compresses a [`ListArray`] by narrowing offsets and recursively compressing elements.
    fn compress_list_array(
        &self,
        list_array: ListArray,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let list_array = list_array.reset_offsets(true)?;

        let compressed_elems = self.compress(list_array.elements())?;

        // Record the root scheme with the offsets child index so root exclusion rules apply.
        let offset_ctx = ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_array.offsets().to_primitive().narrow()?),
            offset_ctx,
        )?;

        Ok(
            ListArray::try_new(compressed_elems, compressed_offsets, list_array.validity())?
                .into_array(),
        )
    }

    /// Compresses a [`ListViewArray`] by narrowing offsets/sizes and recursively compressing
    /// elements.
    fn compress_list_view_array(
        &self,
        list_view: ListViewArray,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let compressed_elems = self.compress(list_view.elements())?;

        let offset_ctx = ctx
            .clone()
            .descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_view.offsets().to_primitive().narrow()?),
            offset_ctx,
        )?;

        let sizes_ctx = ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::SIZES);
        let compressed_sizes = self.compress_canonical(
            Canonical::Primitive(list_view.sizes().to_primitive().narrow()?),
            sizes_ctx,
        )?;

        Ok(ListViewArray::try_new(
            compressed_elems,
            compressed_offsets,
            compressed_sizes,
            list_view.validity(),
        )?
        .into_array())
    }
}

/// Returns `true` if `ratio` is a valid compression ratio (> 1.0, finite, not subnormal) that
/// beats the current best.
fn is_better_ratio(ratio: f64, best: &Option<(&'static dyn Scheme, f64)>) -> bool {
    ratio.is_finite() && !ratio.is_subnormal() && ratio > 1.0 && best.is_none_or(|(_, r)| ratio > r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::FloatDictScheme;
    use crate::builtins::IntDictScheme;
    use crate::builtins::StringDictScheme;
    use crate::ctx::CompressorContext;
    use crate::scheme::SchemeExt;

    fn compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![&IntDictScheme, &FloatDictScheme, &StringDictScheme])
    }

    #[test]
    fn test_self_exclusion() {
        let c = compressor();
        let ctx = CompressorContext::default().descend_with_scheme(IntDictScheme.id(), 0);

        // IntDictScheme is in the history, so it should be excluded.
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_root_exclusion_list_offsets() {
        let c = compressor();
        let ctx = CompressorContext::default()
            .descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);

        // IntDict should be excluded for list offsets.
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_push_rule_float_dict_excludes_int_dict_from_codes() {
        let c = compressor();
        // FloatDict cascading through codes (child 1).
        let ctx = CompressorContext::default().descend_with_scheme(FloatDictScheme.id(), 1);

        // IntDict should be excluded from FloatDict's codes child.
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_push_rule_float_dict_excludes_int_dict_from_values() {
        let c = compressor();
        // FloatDict cascading through values (child 0).
        let ctx = CompressorContext::default().descend_with_scheme(FloatDictScheme.id(), 0);

        // IntDict should also be excluded from FloatDict's values child (ALP propagation
        // replacement).
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_no_exclusion_without_history() {
        let c = compressor();
        let ctx = CompressorContext::default();

        // No history means no exclusions.
        assert!(!c.is_excluded(&IntDictScheme, &ctx));
    }
}
