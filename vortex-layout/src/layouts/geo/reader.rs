// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use geo_types::Geometry;
use geozero::geo_types::GeoWriter;
use geozero::wkb;
use geozero::GeozeroGeometry;
use itertools::Itertools;
use vortex_array::expr::root;
use vortex_array::expr::st_contains::STContains;
use vortex_array::expr::Expression;
use vortex_array::expr::Literal;
use vortex_array::Array;
use vortex_array::MaskFuture;
use vortex_buffer::BitBufferMut;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::layouts::geo::GeoFilter;
use crate::layouts::geo::GeoLayout;
use crate::layouts::geo::SharedGeoFilter;
use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;

pub struct GeoReader {
    pub(crate) name: Arc<str>,
    pub(crate) layout: GeoLayout,
    pub(crate) children: LazyReaderChildren,
    pub(crate) geo_filter: OnceLock<SharedGeoFilter>,
    // TODO(aduffy): cache the pruning result
    // pub(crate) pruning_result: LazyLock<DashMap<Expression, Option>>,
}

impl GeoReader {
    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.children.get(0)
    }

    /// Get the range of zone IDs containing a row range.
    pub(crate) fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        // Zone length is guaranteed to be > 0 by ZonedLayout::new validation
        debug_assert!(self.layout.zone_len > 0, "zone_len must be > 0");
        let zone_len_u64 = self.layout.zone_len as u64;
        let zone_start = row_range.start / zone_len_u64;
        let zone_end = row_range.end.div_ceil(zone_len_u64);
        zone_start..zone_end
    }

    /// Get the row index for the first row in a zone with the given `zone_index`.
    pub(crate) fn first_row_offset(&self, zone_idx: u64) -> u64 {
        zone_idx
            .saturating_mul(self.layout.zone_len as u64)
            .min(self.layout.row_count())
    }

    fn geo_filter(&self) -> SharedGeoFilter {
        self.geo_filter
            .get_or_init(move || {
                let nzones = self.layout.nzones();

                let zones_eval = self
                    .children
                    .get(1)
                    .vortex_expect("failed to get zone child")
                    .projection_evaluation(
                        &(0..nzones as u64),
                        &root(),
                        MaskFuture::new_true(nzones),
                    )
                    .vortex_expect("Failed construct zone map evaluation");

                async move {
                    let zones_array = zones_eval.await?;
                    GeoFilter::try_load(zones_array)
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }
}

impl LayoutReader for GeoReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.layout.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // Register splits from the data.
        self.children
            .get(0)?
            .register_splits(field_mask, row_range, splits)?;

        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let data_eval = self
            .data_child()?
            .pruning_evaluation(row_range, expr, mask.clone())?;

        if let Some(st_contains) = expr.as_opt::<STContains>() {
            // Get the st_contains by scanning the input
            let lhs = st_contains.child(0);
            let rhs = st_contains.child(1);

            if let Some(lhs_lit) = lhs.as_opt::<Literal>() {
                // Return the applied version from this instead.
                let Some(v) = lhs_lit.data().as_binary_opt() else {
                    return Ok(MaskFuture::ready(mask));
                };

                let Some(wkb) = v.value() else {
                    return Ok(MaskFuture::ready(mask));
                };

                let geometry = parse_wkb(&wkb);

                // Get the literal from it here.
                let geo_filter = self.geo_filter();

                // Append the new mask instead here
                let len = mask.len();
                let zone_len = self.layout.zone_len;
                let row_count = row_range.end - row_range.start;
                let zone_range = self.zone_range(row_range);
                let zone_lengths: Vec<_> = zone_range
                    .clone()
                    .map(|zone_idx| {
                        // Figure out the range in the mask that corresponds to the zone
                        let start = usize::try_from(
                            self.first_row_offset(zone_idx)
                                .saturating_sub(row_range.start),
                        )?;
                        let end = usize::try_from(
                            self.first_row_offset(zone_idx + 1)
                                .saturating_sub(row_range.start)
                                .min(row_count),
                        )?;
                        Ok::<_, VortexError>(end - start)
                    })
                    .try_collect()?;

                return Ok(MaskFuture::new(len, async move {
                    let geo_filter = geo_filter.await?;
                    let mut builder = BitBufferMut::with_capacity(mask.len());
                    for (zone_idx, &zone_length) in zone_range.clone().zip_eq(&zone_lengths) {
                        builder.append_n(
                            !geo_filter.filter_contains(usize::try_from(zone_idx)?, &geometry),
                            zone_length,
                        );
                    }

                    let stats_mask = Mask::from(builder.freeze());
                    assert_eq!(stats_mask.len(), mask.len(), "Mask length mismatch");

                    // Intersect the masks.
                    let mut stats_mask = mask.bitand(&stats_mask);

                    // Forward to data child for further pruning.
                    if !stats_mask.all_false() {
                        let data_mask = data_eval.await?;
                        stats_mask = stats_mask.bitand(&data_mask);
                    }

                    Ok(stats_mask)
                }));
            }
        }

        // Re-run the data eval.
        self.data_child()?.pruning_evaluation(row_range, expr, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.data_child()?.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // TODO(aduffy): can we do anything better here?
        self.data_child()?
            .projection_evaluation(row_range, expr, mask)
    }
}

fn parse_wkb(wkb: &[u8]) -> Geometry {
    let mut writer = GeoWriter::new();
    wkb::Wkb(wkb)
        .process_geom(&mut writer)
        .expect("wkb parsing left");
    writer.take_geometry().expect("wkb should yield geometry")
}
