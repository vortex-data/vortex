// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, OnceLock};

use crate::LayoutReader;
use crate::segments::SegmentSource;
use arrow_buffer::BooleanBufferBuilder;
use fastbloom::BloomFilter;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
use vortex_array::Canonical;
use vortex_array::stats::Precision;
use vortex_array::{ArrayRef, MaskFuture};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{SharedVortexResult, VortexExpect, VortexResult, vortex_bail};
use vortex_expr::{
    BinaryVTable, ExprRef, ListContainsVTable, LiteralVTable, Operator, is_root, root,
};
use vortex_mask::Mask;
use vortex_utils::aliases::dash_map::DashMap;

use super::{BloomZonedLayout, deserialize_bloom};

/// Holds the cached pruning mask derived from the bloom filters for a
/// particular predicate.
struct BloomPruningResult {
    mask: Mask,
}

impl BloomPruningResult {
    fn mask(&self) -> VortexResult<Mask> {
        Ok(self.mask.clone())
    }
}

type BloomFuture = BoxFuture<'static, SharedVortexResult<Arc<Vec<BloomFilter>>>>;
type SharedBlooms = Shared<BloomFuture>;

type BloomPruningFuture = BoxFuture<'static, SharedVortexResult<Arc<BloomPruningResult>>>;
type SharedPruningResult = Shared<BloomPruningFuture>;

pub struct BloomZonedReader {
    /// Layout containing the child data layout and bloom metadata layout.
    layout: BloomZonedLayout,
    /// Fully-qualified layout name used for diagnostics.
    name: Arc<str>,
    /// Reader for the wrapped data layout.
    data_child: Arc<dyn LayoutReader>,
    /// Reader that exposes the serialized bloom filters.
    bloom_child: Arc<dyn LayoutReader>,
    /// Lazily materialized bloom filter collection shared across evaluations.
    blooms: OnceLock<SharedBlooms>,
    /// Cache of pruning futures keyed by predicate expression.
    pruning_results: DashMap<ExprRef, Option<SharedPruningResult>>,
}

impl BloomZonedReader {
    pub(super) fn try_new(
        layout: BloomZonedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<Self> {
        let data_child = layout
            .data()
            .new_reader(name.clone(), segment_source.clone())?;
        let bloom_child = layout
            .bloom_zones()
            .new_reader(format!("{name}.bloom_zones").into(), segment_source)?;

        Ok(Self {
            layout,
            name,
            data_child,
            bloom_child,
            blooms: OnceLock::new(),
            pruning_results: Default::default(),
        })
    }

    /// Materialize the bloom filters exactly once, caching the future behind an
    /// [`OnceLock`]. The underlying layout requires a projection to retrieve the
    /// serialized bloom blobs, which we decode into concrete `BloomFilter`
    /// instances. Subsequent callers clone the shared future instead of
    /// repeating IO work.
    fn blooms(&self) -> SharedBlooms {
        self.blooms
            .get_or_init(|| {
                let nzones = self.layout.nzones();
                let seed = self.layout.seed();
                let bloom_eval = self
                    .bloom_child
                    .projection_evaluation(
                        &(0..nzones as u64),
                        &root(),
                        MaskFuture::new_true(nzones),
                    )
                    .vortex_expect("Failed to construct bloom zones projection");

                async move {
                    let zones_array = bloom_eval.await?;
                    let canonical = zones_array.to_canonical();
                    let Canonical::VarBinView(view) = canonical else {
                        vortex_bail!(
                            "Bloom zones layout produced non-binary data: {:?}",
                            zones_array.dtype()
                        );
                    };

                    if view.len() != nzones {
                        vortex_bail!(
                            "Bloom zones length mismatch: expected {nzones}, got {}",
                            view.len()
                        );
                    }

                    let mut blooms = Vec::with_capacity(nzones);
                    for idx in 0..nzones {
                        let bytes = view.bytes_at(idx);
                        let bloom = deserialize_bloom(bytes.as_slice(), seed)?;
                        blooms.push(bloom);
                    }

                    Ok(Arc::new(blooms))
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }

    /// Return a shared future that evaluates the pruning mask for the supplied
    /// predicate, if the predicate can be satisfied via bloom evaluation.
    fn pruning_mask_future(&self, expr: ExprRef) -> Option<SharedPruningResult> {
        self.pruning_results
            .entry(expr.clone())
            .or_insert_with(|| {
                let values = extract_pruning_values(&expr)?;
                let blooms = self.blooms();
                Some(
                    async move {
                        let blooms = blooms.await?;
                        let mask = compute_pruning_mask(&blooms, &values)?;
                        Ok(Arc::new(BloomPruningResult { mask }))
                    }
                    .map_err(Arc::new)
                    .boxed()
                    .shared(),
                )
            })
            .clone()
    }

    fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        let zone_len = self.layout.zone_len() as u64;
        let start = row_range.start / zone_len;
        let end = row_range.end.div_ceil(zone_len);
        start..end
    }

    fn first_row_offset(&self, zone_idx: u64) -> u64 {
        zone_idx
            .saturating_mul(self.layout.zone_len() as u64)
            .min(self.layout.data().row_count())
    }
}

impl LayoutReader for BloomZonedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.data_child.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        self.data_child.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.data_child
            .register_splits(field_mask, row_offset, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let data_eval = self
            .data_child
            .pruning_evaluation(row_range, expr, mask.clone())?;

        let Some(pruning_future) = self.pruning_mask_future(expr.clone()) else {
            return Ok(data_eval);
        };

        let row_count = row_range.end - row_range.start;
        let zone_indices: Vec<u64> = self.zone_range(row_range).collect();
        let mut zone_lengths = Vec::with_capacity(zone_indices.len());
        for &zone_idx in &zone_indices {
            // Translate the row range into per-zone lengths. The pruning mask is
            // maintained per zone, so we expand each zone decision into the
            // number of rows it contributes within the requested range.
            let start_offset = self
                .first_row_offset(zone_idx)
                .saturating_sub(row_range.start);
            let end_offset = self
                .first_row_offset(zone_idx + 1)
                .saturating_sub(row_range.start)
                .min(row_count);
            let length = usize::try_from(end_offset.saturating_sub(start_offset))?;
            zone_lengths.push(length);
        }

        let zone_indices_for_future = zone_indices;
        let zone_lengths_for_future = zone_lengths;

        Ok(MaskFuture::new(mask.len(), async move {
            let pruning_mask = pruning_future.clone().await?.mask()?;

            let mut builder = BooleanBufferBuilder::new(mask.len());
            for (zone_idx, zone_length) in zone_indices_for_future
                .iter()
                .zip(zone_lengths_for_future.iter())
            {
                let prune_zone = pruning_mask.value(usize::try_from(*zone_idx)?);
                builder.append_n(*zone_length, !prune_zone);
            }

            let stats_mask = Mask::from(vortex_buffer::BitBuffer::from(builder.finish()));
            let mut combined = mask.bitand(&stats_mask);

            if !combined.all_false() {
                let data_mask = data_eval.await?;
                combined = combined.bitand(&data_mask);
            }

            Ok(combined)
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.data_child.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        self.data_child.projection_evaluation(row_range, expr, mask)
    }
}

/// Extract string literals from predicates supported by bloom pruning.
fn extract_pruning_values(expr: &ExprRef) -> Option<Vec<String>> {
    if let Some(binary) = expr.as_opt::<BinaryVTable>()
        && binary.op() == Operator::Eq
    {
        if is_root(binary.lhs()) {
            return literal_utf8(binary.rhs()).map(|value| vec![value]);
        }
        if is_root(binary.rhs()) {
            return literal_utf8(binary.lhs()).map(|value| vec![value]);
        }
    }

    if expr.is::<ListContainsVTable>() {
        let children = expr.children();
        if children.len() == 2 && is_root(children[1]) {
            return literal_utf8_list(children[0]);
        }
    }

    None
}

/// Return the UTF-8 string value if the expression is a literal.
fn literal_utf8(expr: &ExprRef) -> Option<String> {
    let literal = expr.as_opt::<LiteralVTable>()?;
    let utf8 = literal.value().as_utf8_opt()?;
    let value = utf8.value()?;
    Some(value.as_str().to_owned())
}

/// Return the UTF-8 string list if the expression is a literal list.
fn literal_utf8_list(expr: &ExprRef) -> Option<Vec<String>> {
    let literal = expr.as_opt::<LiteralVTable>()?;
    let list = literal.value().as_list_opt()?;
    let elements = list.elements()?;
    let mut values = Vec::with_capacity(elements.len());
    for scalar in elements {
        let utf8 = scalar.as_utf8_opt()?;
        let value = utf8.value()?;
        values.push(value.as_str().to_owned());
    }
    Some(values)
}

/// Compute the pruning mask for the provided bloom filters and literal values.
fn compute_pruning_mask(blooms: &[BloomFilter], values: &[String]) -> VortexResult<Mask> {
    if blooms.is_empty() {
        return Ok(Mask::new_false(0));
    }

    if values.is_empty() {
        return Ok(Mask::new_true(blooms.len()));
    }

    let mut builder = BooleanBufferBuilder::new(blooms.len());
    for bloom in blooms {
        let maybe_match = values.iter().any(|value| bloom.contains(value.as_str()));
        builder.append(!maybe_match);
    }

    Ok(Mask::from(vortex_buffer::BitBuffer::from(builder.finish())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vortex_dtype::{DType, Nullability};
    use vortex_expr::{eq, gt, list_contains, lit, root};
    use vortex_scalar::Scalar;

    #[test]
    fn extract_eq_literal() {
        let expr = eq(root(), lit("value"));
        let values = extract_pruning_values(&expr).expect("values");
        assert_eq!(values, vec!["value".to_string()]);
    }

    #[test]
    fn extract_list_contains() {
        let list_scalar = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![Scalar::from("a"), Scalar::from("b")],
            Nullability::NonNullable,
        );
        let expr = list_contains(lit(list_scalar), root());
        let values = extract_pruning_values(&expr).expect("values");
        assert_eq!(values, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn extract_eq_literal_reversed() {
        // Test literal = root() (reversed order)
        let expr = eq(lit("value"), root());
        let values = extract_pruning_values(&expr).expect("values");
        assert_eq!(values, vec!["value".to_string()]);
    }

    #[test]
    fn extract_unsupported_returns_none() {
        // Non-equality operator (not supported)
        let expr = gt(root(), lit("value"));
        assert!(extract_pruning_values(&expr).is_none());

        // Literal compared to literal (no root)
        let expr = eq(lit("a"), lit("b"));
        assert!(extract_pruning_values(&expr).is_none());

        // Non-UTF8 literal
        let expr = eq(root(), lit(42i32));
        assert!(extract_pruning_values(&expr).is_none());
    }

    #[test]
    fn extract_list_contains_wrong_order() {
        // list_contains with root in wrong position
        let list_scalar = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![Scalar::from("a"), Scalar::from("b")],
            Nullability::NonNullable,
        );
        // root() in first position instead of second - not supported
        let expr = list_contains(root(), lit(list_scalar));
        assert!(extract_pruning_values(&expr).is_none());
    }

    #[test]
    fn compute_pruning_mask_values() {
        let mut bloom = BloomFilter::with_false_pos(0.01)
            .seed(&10)
            .expected_items(2);
        bloom.insert("match");

        let bloom2 = BloomFilter::with_false_pos(0.01)
            .seed(&10)
            .expected_items(2);
        let mask = compute_pruning_mask(&[bloom, bloom2], &["match".into()]).unwrap();

        assert_eq!(mask.len(), 2);
        assert!(!mask.value(0));
        assert!(mask.value(1));
    }
}
