// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A writer strategy for struct-typed arrays.
//!
//! [`StructStrategy`] transposes a stream of struct chunks into one ordered stream per field
//! (plus a validity stream when the struct is nullable) and writes each through a configurable
//! child strategy, producing a single [`StructLayout`]. It is a *structural* writer: it does not
//! inspect child dtypes or resolve field-path overrides itself. Dispatching a child to the right
//! layout kind is the job of the caller (see [`TableStrategy`]).
//!
//! [`TableStrategy`]: crate::layouts::table::TableStrategy

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join;
use futures::future::try_join_all;
use futures::pin_mut;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::DemandChannel;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Writes struct-typed arrays into a [`StructLayout`], one child layout per field.
///
/// Each field is written through a strategy resolved by direct field name: an entry in
/// `field_writers` if present, otherwise `default`. When the struct is nullable, its validity
/// bitmap is written through `validity`.
///
/// `StructStrategy` is intentionally unaware of nested dtypes and field-path overrides. To write
/// arbitrarily nested struct trees with per-path overrides, drive it from
/// [`TableStrategy`][crate::layouts::table::TableStrategy], which dispatches on dtype and resolves
/// the per-field child strategies before handing them here.
#[derive(Clone)]
pub struct StructStrategy {
    /// Per-field child strategies, keyed by direct field name. Fields without an entry use
    /// `default`.
    field_writers: HashMap<FieldName, Arc<dyn LayoutStrategy>>,
    /// Strategy for fields that have no entry in `field_writers`.
    default: Arc<dyn LayoutStrategy>,
    /// Strategy for the struct's own validity bitmap, used only when the struct is nullable.
    validity: Arc<dyn LayoutStrategy>,
}

impl StructStrategy {
    /// Create a new struct writer that writes every field through `default` and the validity
    /// bitmap (when present) through `validity`.
    pub fn new(validity: Arc<dyn LayoutStrategy>, default: Arc<dyn LayoutStrategy>) -> Self {
        Self {
            field_writers: HashMap::default(),
            default,
            validity,
        }
    }

    /// Override the strategy for a single field by name.
    pub fn with_field_writer(
        mut self,
        name: impl Into<FieldName>,
        writer: Arc<dyn LayoutStrategy>,
    ) -> Self {
        self.field_writers.insert(name.into(), writer);
        self
    }

    /// Override the strategy for several fields by name at once.
    pub fn with_field_writers(
        mut self,
        writers: impl IntoIterator<Item = (FieldName, Arc<dyn LayoutStrategy>)>,
    ) -> Self {
        self.field_writers.extend(writers);
        self
    }
}

#[async_trait]
impl LayoutStrategy for StructStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
            vortex_bail!("StructStrategy can only write struct-typed streams, got {dtype}");
        };

        // Check for unique field names at write time.
        if HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
            != struct_dtype.names().len()
        {
            vortex_bail!("StructLayout must have unique field names");
        }
        let is_nullable = dtype.is_nullable();

        // Optimization: when there are no fields, don't spawn any work and just write a trivial
        // StructLayout.
        if struct_dtype.nfields() == 0 && !is_nullable {
            let row_count = stream
                .try_fold(
                    0u64,
                    |acc, (_, arr)| async move { Ok(acc + arr.len() as u64) },
                )
                .await?;
            return Ok(StructLayout::new(row_count, dtype, vec![]).into_layout());
        }

        // Fields are written unmasked, so the struct publishes which of their rows the logical
        // fields contain: the rows its own ancestors demand where the struct is valid. A struct
        // under a nullable ancestor whose demand it cannot claim cannot derive its fields' demand.
        let upstream_demand = ctx.claim_demand();
        let field_demands: Option<Vec<DemandChannel>> = (upstream_demand.is_some()
            || (is_nullable && !ctx.has_nullable_ancestor()))
        .then(|| {
            (0..struct_dtype.nfields())
                .map(|_| DemandChannel::new())
                .collect()
        });

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_session = session.clone();
        let publish_demands = field_demands.clone();
        let columns_vec_stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let mut ctx = columns_session.create_execution_ctx();
            let struct_chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
            let mut columns: Vec<(SequenceId, ArrayRef)> = Vec::new();
            let validity = is_nullable
                .then(|| chunk.validity()?.execute_mask(chunk.len(), &mut ctx))
                .transpose()?;

            // Publish the fields' demand before sending the fields, so their writers find it.
            if let Some(publish_demands) = &publish_demands {
                let parent_demand = match &upstream_demand {
                    Some(upstream_demand) => upstream_demand.take(chunk.len()),
                    None => Some(Mask::new_true(chunk.len())),
                };
                match parent_demand {
                    Some(parent_demand) => {
                        let demand = match &validity {
                            Some(validity) => &parent_demand & validity,
                            None => parent_demand,
                        };
                        for channel in publish_demands {
                            channel.push(demand.clone());
                        }
                    }
                    None => publish_demands.iter().for_each(DemandChannel::lose),
                }
            }

            if let Some(validity) = validity {
                columns.push((sequence_pointer.advance(), validity.into_array()));
            }

            columns.extend(
                struct_chunk
                    .iter_unmasked_fields()
                    .map(|field| (sequence_pointer.advance(), field.clone())),
            );

            Ok(columns)
        });

        let mut stream_count = struct_dtype.nfields();
        if is_nullable {
            stream_count += 1;
        }

        let (column_streams_tx, column_streams_rx): (Vec<_>, Vec<_>) =
            (0..stream_count).map(|_| kanal::bounded_async(1)).unzip();

        // Fan out column chunks to their respective transposed streams. Keep this future joined
        // with the column writers so producer panics/errors cannot be hidden as channel EOF.
        let handle = session.handle();
        let fanout_fut = async move {
            pin_mut!(columns_vec_stream);
            while let Some(result) = columns_vec_stream.next().await {
                match result {
                    Ok(columns) => {
                        for (tx, column) in column_streams_tx.iter().zip_eq(columns) {
                            if tx.send(Ok(column)).await.is_err() {
                                vortex_bail!(
                                    "struct column writer finished before all chunks were sent"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        let e: Arc<VortexError> = Arc::new(e);
                        for tx in column_streams_tx.iter() {
                            let _ = tx.send(Err(VortexError::from(Arc::clone(&e)))).await;
                        }
                        return Err(VortexError::from(e));
                    }
                }
            }
            Ok(())
        };

        // First child column is the validity, subsequent children are the individual struct fields
        let column_dtypes: Vec<DType> = if is_nullable {
            std::iter::once(DType::Bool(Nullability::NonNullable))
                .chain(struct_dtype.fields())
                .collect()
        } else {
            struct_dtype.fields().collect()
        };

        let column_names: Vec<FieldName> = if is_nullable {
            std::iter::once(FieldName::from("__validity"))
                .chain(struct_dtype.names().iter().cloned())
                .collect()
        } else {
            struct_dtype.names().iter().cloned().collect()
        };

        let layout_futures: Vec<_> = column_dtypes
            .into_iter()
            .zip_eq(column_streams_rx)
            .zip_eq(column_names)
            .enumerate()
            .map(move |(index, ((dtype, recv), name))| {
                let column_stream =
                    SequentialStreamAdapter::new(dtype, recv.into_stream().boxed()).sendable();
                let child_eof = eof.split_off();
                let session = session.clone();
                let is_validity = index == 0 && is_nullable;
                // Fields are written unmasked, so a nullable struct hides some of their values
                // behind its own validity. Hand the field writers the demand on their rows, or,
                // without it, tell them so they omit statistics those hidden values would corrupt.
                let ctx = ctx.clone().without_demand();
                let ctx = if is_validity {
                    ctx
                } else if let Some(field_demands) = &field_demands {
                    ctx.with_demand(field_demands[index - usize::from(is_nullable)].clone())
                } else if is_nullable {
                    ctx.with_nullable_ancestor()
                } else {
                    ctx
                };
                let segment_sink = Arc::clone(&segment_sink);
                handle.spawn_nested(move |_| {
                    // Validity is written through the validity strategy; every other field
                    // resolves to its named override or the default strategy.
                    let writer = if is_validity {
                        Arc::clone(&self.validity)
                    } else {
                        self.field_writers
                            .get(&name)
                            .cloned()
                            .unwrap_or_else(|| Arc::clone(&self.default))
                    };

                    async move {
                        writer
                            .write_stream(ctx, segment_sink, column_stream, child_eof, &session)
                            .await
                    }
                })
            })
            .collect();

        let (_success, column_layouts) = try_join(fanout_fut, try_join_all(layout_futures)).await?;
        // TODO(os): transposed stream could count row counts as well,
        // This must hold though, all columns must have the same row count of the struct layout
        let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
        Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use vortex_array::ArrayContext;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::arrays::BoolArray;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::writer::ZonedLayoutOptions;
    use crate::layouts::zoned::writer::ZonedStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;
    use crate::test::read_zone_stats;

    fn zoned_leaf() -> Arc<dyn LayoutStrategy> {
        Arc::new(ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        ))
    }

    /// Write `array` through `strategy`, returning the layout and the segments it was written to.
    fn write<S: LayoutStrategy>(
        strategy: S,
        array: ArrayRef,
    ) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let segments = Arc::new(TestSegments::default());
        let segments2 = Arc::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    LayoutWriterContext::new(ArrayContext::empty()),
                    segments2,
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;
        Ok((layout, segments))
    }

    /// Write a single-field struct of `[1, 2, 3]` with zoned fields, returning the field's layout.
    fn write_field(validity: Validity) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
        let strategy = StructStrategy::new(Arc::new(FlatLayoutStrategy::default()), zoned_leaf());
        let is_nullable = validity.nullability().is_nullable();
        let array = StructArray::try_from_iter_with_validity(
            [("a", buffer![1, 2, 3].into_array())],
            validity,
        )?;
        let (layout, segments) = write(strategy, array.into_array())?;
        // The validity child, when present, precedes the fields.
        Ok((
            Arc::clone(&layout.children()?[usize::from(is_nullable)]),
            segments,
        ))
    }

    /// Write `array` through a [`TableStrategy`] with zoned leaves.
    fn write_table(array: StructArray) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
        let strategy = TableStrategy::new(Arc::new(FlatLayoutStrategy::default()), zoned_leaf());
        write(strategy, array.into_array())
    }

    /// The minimum and maximum a zoned leaf of at most three rows records for its single zone.
    fn zone_bounds(leaf: &LayoutRef, segments: &Arc<TestSegments>) -> VortexResult<(i32, i32)> {
        assert!(leaf.is::<Zoned>(), "{leaf} has no zone map");
        let mut ctx = new_session().create_execution_ctx();
        let mut bound = |aggregate_fn| -> VortexResult<i32> {
            let scalar: Scalar = read_zone_stats(leaf, segments.clone(), &aggregate_fn)?
                .execute_scalar(0, &mut ctx)?;
            Ok(scalar
                .as_primitive()
                .typed_value::<i32>()
                .vortex_expect("demanded zone has a bound"))
        };
        Ok((
            bound(Min.bind(Default::default()))?,
            bound(Max.bind(Default::default()))?,
        ))
    }

    fn some_nulls() -> Validity {
        Validity::Array(BoolArray::from_iter([false, true, true]).into_array())
    }

    #[test]
    fn non_nullable_struct_fields_get_a_zone_map() -> VortexResult<()> {
        let (field, segments) = write_field(Validity::NonNullable)?;
        assert_eq!(zone_bounds(&field, &segments)?, (1, 3));
        Ok(())
    }

    /// The field holds a value in the row the struct nulls out, and its zone map leaves it out.
    #[test]
    fn nullable_struct_fields_zone_map_only_valid_rows() -> VortexResult<()> {
        let (field, segments) = write_field(some_nulls())?;
        assert_eq!(zone_bounds(&field, &segments)?, (2, 3));
        Ok(())
    }

    /// A nullable struct anywhere above a leaf hides its values, however deep the leaf sits, while
    /// leaves beside a nullable struct are unaffected.
    #[test]
    fn demand_propagates_through_nested_structs() -> VortexResult<()> {
        // Nullable outer struct, non-nullable inner struct: the inner leaf still skips row 0.
        let outer_nullable = StructArray::try_from_iter_with_validity(
            [(
                "inner",
                StructArray::try_from_iter([("a", buffer![1, 2, 3].into_array())])?.into_array(),
            )],
            some_nulls(),
        )?;
        let (layout, segments) = write_table(outer_nullable)?;
        let inner = &layout.children()?[1];
        assert_eq!(zone_bounds(&inner.children()?[0], &segments)?, (2, 3));

        // Non-nullable outer struct: a leaf beside the nullable inner struct covers every row,
        // the leaf below it skips row 0.
        let inner_nullable = StructArray::try_from_iter([
            ("sibling", buffer![1, 2, 3].into_array()),
            (
                "inner",
                StructArray::try_from_iter_with_validity(
                    [("a", buffer![1, 2, 3].into_array())],
                    some_nulls(),
                )?
                .into_array(),
            ),
        ])?;
        let (layout, segments) = write_table(inner_nullable)?;
        let children = layout.children()?;
        assert_eq!(zone_bounds(&children[0], &segments)?, (1, 3));
        assert_eq!(zone_bounds(&children[1].children()?[1], &segments)?, (2, 3));
        Ok(())
    }

    /// The demand on a leaf under two nullable structs is the rows where both are valid.
    #[test]
    fn nested_nullable_structs_compose_their_validity() -> VortexResult<()> {
        let inner = StructArray::try_from_iter_with_validity(
            [("a", buffer![1, 2, 3].into_array())],
            some_nulls(),
        )?;
        let outer = StructArray::try_from_iter_with_validity(
            [("inner", inner.into_array())],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        )?;
        let (layout, segments) = write_table(outer)?;
        let leaf = &layout.children()?[1].children()?[1];
        assert_eq!(zone_bounds(leaf, &segments)?, (2, 2));
        Ok(())
    }

    /// Wrap `leaf` in `depth` non-nullable single-field structs named `l0`, `l1`, ...
    fn nest(leaf: ArrayRef, depth: usize) -> VortexResult<ArrayRef> {
        (0..depth).try_fold(leaf, |inner, level| {
            Ok(StructArray::try_from_iter([(format!("l{level}"), inner)])?.into_array())
        })
    }

    /// Descend `depth` single-field non-nullable struct layouts to their leaf.
    fn descend(layout: &LayoutRef, depth: usize) -> VortexResult<LayoutRef> {
        (0..depth).try_fold(Arc::clone(layout), |layout, _| {
            Ok(Arc::clone(&layout.children()?[0]))
        })
    }

    /// The demand survives any number of non-nullable struct levels below the nullable one, and
    /// only applies below the nullable level, not above it.
    #[test]
    fn demand_propagates_through_deep_nesting() -> VortexResult<()> {
        const DEPTH: usize = 6;

        // Nullable root over DEPTH non-nullable levels.
        let root = StructArray::try_from_iter_with_validity(
            [("top", nest(buffer![1, 2, 3].into_array(), DEPTH)?)],
            some_nulls(),
        )?;
        let (layout, segments) = write_table(root)?;
        // Skip the root's validity child, then walk every non-nullable level to the leaf.
        let leaf = descend(&layout.children()?[1], DEPTH)?;
        assert_eq!(zone_bounds(&leaf, &segments)?, (2, 3));

        // Non-nullable levels above, a nullable struct in the middle, non-nullable levels below.
        // A leaf above the nullable level covers every row, the leaf below it skips row 0.
        let middle = StructArray::try_from_iter_with_validity(
            [("below", nest(buffer![1, 2, 3].into_array(), DEPTH)?)],
            some_nulls(),
        )?;
        let root = nest(
            StructArray::try_from_iter([
                ("above", buffer![1, 2, 3].into_array()),
                ("middle", middle.into_array()),
            ])?
            .into_array(),
            DEPTH,
        )?
        .execute::<StructArray>(&mut new_session().create_execution_ctx())?;
        let (layout, segments) = write_table(root)?;
        let holder = descend(&layout, DEPTH)?;
        let holder_children = holder.children()?;
        assert_eq!(zone_bounds(&holder_children[0], &segments)?, (1, 3));
        // The nullable middle struct's children are `[validity, below]`.
        let leaf = descend(&holder_children[1].children()?[1], DEPTH)?;
        assert_eq!(zone_bounds(&leaf, &segments)?, (2, 3));
        Ok(())
    }
}
