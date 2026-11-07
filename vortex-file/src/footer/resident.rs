// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Estimates of the heap memory a parsed footer keeps alive, which weigh it in footer caches.
//!
//! The layout tree is a view over a retained flatbuffer whose nodes materialize as they are visited
//! and then stay cached on the footer. The estimate counts every node as materialized, as it is
//! once a scan has read every column, but walks the flatbuffer so that estimating materializes
//! nothing. The dtype's share comes from [`DType::approx_resident_bytes`]. The per-node constants
//! were measured with a counting allocator.
//!
//! [`DType::approx_resident_bytes`]: vortex_array::dtype::DType::approx_resident_bytes

use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_layout::flatbuffers::layout as fbl;
use vortex_layout::layouts::file_stats::AggregateStat;

use crate::footer::FileStatistics;
use crate::footer::SegmentSpec;

/// A materialized layout node with its child adapter, segment ids and encoding-specific data.
const LAYOUT_NODE_BYTES: usize = 680;
/// A statistics entry without its aggregates: its dtype, its path twice and its path index slot.
const STATS_ENTRY_BYTES: usize = 160;

/// Estimates the bytes a parsed footer holds once every dtype and layout node is materialized.
///
/// `dtype_bytes` is the dtype's share, from [`DType::approx_resident_bytes`].
///
/// [`DType::approx_resident_bytes`]: vortex_array::dtype::DType::approx_resident_bytes
pub(super) fn footer_resident_bytes(
    dtype_bytes: usize,
    layout_flatbuffer: &[u8],
    segments: usize,
    statistics: Option<&FileStatistics>,
    metadata_bytes: usize,
) -> usize {
    // SAFETY: the caller has parsed `layout_flatbuffer` into a layout, which verified it.
    let layout = unsafe { flatbuffers::root_unchecked::<fbl::Layout<'_>>(layout_flatbuffer) };
    sum([
        dtype_bytes,
        layout_flatbuffer.len(),
        layout_nodes(layout).saturating_mul(LAYOUT_NODE_BYTES),
        segments.saturating_mul(size_of::<SegmentSpec>()),
        statistics.map_or(0, statistics_bytes),
        metadata_bytes,
    ])
}

fn layout_nodes(layout: fbl::Layout<'_>) -> usize {
    1usize.saturating_add(
        layout
            .children()
            .map_or(0, |children| sum(children.iter().map(layout_nodes))),
    )
}

fn statistics_bytes(statistics: &FileStatistics) -> usize {
    sum(statistics.aggregates().iter().map(|aggregates| {
        STATS_ENTRY_BYTES.saturating_add(sum(aggregates.iter().map(aggregate_bytes)))
    }))
}

fn aggregate_bytes(aggregate: &AggregateStat) -> usize {
    sum([
        size_of::<AggregateStat>(),
        aggregate.partial().map_or(0, scalar_bytes),
        aggregate
            .value()
            .as_ref()
            .into_inner()
            .map_or(0, scalar_bytes),
    ])
}

/// [`Scalar::approx_nbytes`] counts the elements of a list or struct value but not the slots they
/// are held in, which dominate for small elements such as the bytes of a fixed-size binary bound.
fn scalar_bytes(scalar: &Scalar) -> usize {
    scalar
        .approx_nbytes()
        .saturating_add(scalar.value().map_or(0, slot_bytes))
}

fn slot_bytes(value: &ScalarValue) -> usize {
    match value {
        ScalarValue::Tuple(elements) => sum(elements.iter().map(|element| {
            size_of::<Option<ScalarValue>>().saturating_add(element.as_ref().map_or(0, slot_bytes))
        })),
        _ => 0,
    }
}

/// Sums without overflowing, since a malformed file can repeat references to shared nodes.
fn sum(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().fold(0, usize::saturating_add)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::Precision;
    use vortex_layout::layouts::file_stats::AggregateStats;

    use super::*;

    /// Statistics of one fixed-size binary field whose max bound is `len` bytes.
    fn binary_bound_statistics(len: u32) -> FileStatistics {
        let element = Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable));
        let bound = Scalar::fixed_size_list(
            Arc::clone(&element),
            (0..len).map(|_| Scalar::from(0u8)).collect(),
            Nullability::NonNullable,
        );
        let max = AggregateStat::from_value(
            Max.bind(NumericalAggregateOpts::skip_nans()),
            Precision::exact(bound),
        );
        FileStatistics::new(
            Arc::new([AggregateStats::new(vec![max])]),
            Arc::new([DType::FixedSizeList(element, len, Nullability::NonNullable)]),
            Arc::new([FieldPath::root()]),
        )
    }

    #[test]
    fn binary_bound_slots_are_charged() {
        let short = statistics_bytes(&binary_bound_statistics(8));
        let long = statistics_bytes(&binary_bound_statistics(40));
        assert!(
            long >= short + 32 * size_of::<Option<ScalarValue>>(),
            "{short} {long}"
        );
    }
}
