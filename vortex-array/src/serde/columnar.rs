// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Columnar serialization of array trees — a parallel to [`SerializedArray`] that sources
//! its encoding-tree metadata from a columnar representation rather than a flatbuffer.
//!
//! `SerializedArray` parses a per-array-tree flatbuffer (`fba::Array`) and navigates it via
//! offsets baked into the fb. That format lives in the trailing buffer of a `FlatLayout`
//! segment. `ColumnarSerializedArray` is the parallel decode entry point used when the
//! encoding tree is stored as a struct-of-Lists vortex array.
//! The plugin contract — [`ArrayChildren`] plus `plugin.deserialize(dtype, len, metadata,
//!  buffers, children, session)` — doesn't care which source the metadata/buffers/children
//! come from, so this module implements the same decode flow without ever constructing or
//! parsing a flatbuffer.
//!
//! The writer entry point [`serialize_to_columnar_tree`] walks an [`ArrayRef`] in
//! pre-order and produces both the data-segment buffer list (no trailing array node flatbuffer)
//! and a [`ColumnarArrayTree`] capturing the encoding tree, per-node stats, and per-buffer
//! descriptors.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::struct_::StructArrayExt;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::BoolBuilder;
use crate::builders::PrimitiveBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::executor::ExecutionCtx;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::ScalarValue;
use crate::serde::ArrayChildren;
use crate::serde::SerializeOptions;
use crate::session::ArraySessionExt;
use crate::stats::StatsSet;
use crate::validity::Validity;

/// Default menu of stats every [`StatsColumns`] tracks unless the caller picks a custom set.
/// Mirrors the historical `fba::ArrayStats` field list so existing files round-trip.
pub const DEFAULT_STATS: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::Sum,
    Stat::NullCount,
    Stat::NaNCount,
    Stat::UncompressedSizeInBytes,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
];

/// How a [`Stat`] is laid out on disk inside a [`StatsColumns`] struct.
///
/// Each kind expands into one or more columns at the [`StructArray`] level. The kind a
/// [`Stat`] uses is decided by [`stat_column_kind`] (`Min`/`Max` get a value + exact tag,
/// `Sum` is exact-only, counts are u64, flags are bool). This dispatch is the one place
/// the schema's value encoding (currently proto-bytes for value stats) is hardcoded — when
/// stats move to the aggregate-function partial scalars, this enum + dispatch is where the
/// migration happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatColumnKind {
    /// Proto-encoded value bytes + a bool "exact" tag. Two columns: `<stat>` and
    /// `<stat>_exact`.
    BinaryValueWithExact,
    /// Proto-encoded value bytes only (no exactness — `Sum` is exact-only by
    /// construction). One column: `<stat>`.
    BinaryValue,
    /// `u64` typed column.
    U64,
    /// `bool` typed column.
    Bool,
}

fn stat_column_kind(stat: Stat) -> StatColumnKind {
    match stat {
        Stat::Min | Stat::Max => StatColumnKind::BinaryValueWithExact,
        Stat::Sum => StatColumnKind::BinaryValue,
        Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => StatColumnKind::U64,
        Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => StatColumnKind::Bool,
    }
}

/// Append the (field-name, dtype) pairs this stat contributes to the [`StatsColumns`]
/// struct schema. Order matters for [`stats_columns_dtype`] — fields are written in the
/// order produced here.
fn stat_schema_fields(stat: Stat, out: &mut Vec<(FieldName, DType)>) {
    let nullable = Nullability::Nullable;
    match stat_column_kind(stat) {
        StatColumnKind::BinaryValueWithExact => {
            out.push((stat.name().into(), DType::Binary(nullable)));
            out.push((
                format!("{}_exact", stat.name()).into(),
                DType::Bool(nullable),
            ));
        }
        StatColumnKind::BinaryValue => {
            out.push((stat.name().into(), DType::Binary(nullable)));
        }
        StatColumnKind::U64 => {
            out.push((stat.name().into(), DType::Primitive(PType::U64, nullable)));
        }
        StatColumnKind::Bool => {
            out.push((stat.name().into(), DType::Bool(nullable)));
        }
    }
}

/// Build the [`StatsColumns`] struct dtype that backs a set of tracked stats.
///
/// Each `Stat` contributes one or two nullable columns; column names follow the stat's
/// name (`min`, `null_count`, …) plus an `_exact` sibling for value-stats whose
/// exactness is recorded separately. Binary value-stats hold `ScalarValue::to_proto_bytes`
/// blobs (decoded with the array's dtype at read time); `sum` is exact-only so there is
/// no `sum_exact` column.
///
/// The schema is dynamic: passing `&[Stat::Min, Stat::NullCount]` produces a 3-field
/// struct, not the full 11. Files written this way carry their menu in the struct's
/// field names — the reader recovers it without out-of-band coordination.
pub fn stats_columns_dtype(stats: &[Stat]) -> DType {
    let mut fields = Vec::<(FieldName, DType)>::new();
    for &stat in stats {
        stat_schema_fields(stat, &mut fields);
    }
    let (names, dtypes): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
    DType::Struct(
        StructFields::new(names.into(), dtypes),
        Nullability::NonNullable,
    )
}

/// Identify the (stat, role) a field name + dtype represents, for read-side dispatch.
///
/// Returns `None` if the field is not a recognized stat column. Roles:
/// - `Value`: the stat's value column (binary blob for value stats, typed for others).
/// - `Exact`: the bool sibling column tagging whether the value-stat was exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatFieldRole {
    Value,
    Exact,
}

fn parse_stat_field(name: &str, dtype: &DType) -> Option<(Stat, StatFieldRole)> {
    // Look for the `_exact` suffix first to disambiguate from value-stats whose own name
    // contains an underscore (e.g. `null_count`).
    if let Some(prefix) = name.strip_suffix("_exact")
        && let Some(stat) = Stat::all().find(|s| s.name() == prefix)
        && matches!(stat_column_kind(stat), StatColumnKind::BinaryValueWithExact)
        && matches!(dtype, DType::Bool(_))
    {
        return Some((stat, StatFieldRole::Exact));
    }
    let stat = Stat::all().find(|s| s.name() == name)?;
    let expected = match stat_column_kind(stat) {
        StatColumnKind::BinaryValueWithExact | StatColumnKind::BinaryValue => {
            matches!(dtype, DType::Binary(_))
        }
        StatColumnKind::U64 => matches!(dtype, DType::Primitive(PType::U64, _)),
        StatColumnKind::Bool => matches!(dtype, DType::Bool(_)),
    };
    expected.then_some((stat, StatFieldRole::Value))
}

fn is_known_stat_field(name: &str, dtype: &DType) -> bool {
    parse_stat_field(name, dtype).is_some()
}

/// Per-node statistics stored as a [`StructArray`] whose schema is produced by
/// [`stats_columns_dtype`] over a caller-chosen list of [`Stat`]s.
///
/// The schema's field names ARE the manifest of tracked stats — recovering "which stats
/// did this file record?" is just inspecting the struct's field names. Hydration into a
/// typed [`StatsSet`] happens at decode time via [`Self::read`], when the array's dtype
/// is known.
#[derive(Debug, Clone)]
pub struct StatsColumns(StructArray);

impl StatsColumns {
    /// Wrap a [`StructArray`], validating its dtype matches a schema that
    /// [`stats_columns_dtype`] could have produced (i.e., every field is one of the
    /// stat columns this module knows about).
    pub fn new(inner: StructArray) -> VortexResult<Self> {
        let DType::Struct(fields, _) = inner.as_ref().dtype() else {
            vortex_bail!(
                "StatsColumns expected Struct dtype, got {}",
                inner.as_ref().dtype()
            );
        };
        for (name, dtype) in fields.names().iter().zip(fields.fields()) {
            if !is_known_stat_field(name.as_ref(), &dtype) {
                vortex_bail!(
                    "StatsColumns has unknown field '{}' with dtype {}",
                    name,
                    dtype
                );
            }
        }
        Ok(Self(inner))
    }

    /// Reference to underlying [`StructArray`].
    pub fn as_struct(&self) -> &StructArray {
        &self.0
    }

    /// Consume into the underlying [`StructArray`].
    pub fn into_struct(self) -> StructArray {
        self.0
    }

    /// Number of node-rows held by these columns.
    pub fn nrows(&self) -> usize {
        self.0.as_ref().len()
    }

    /// The set of stats this column array tracks, recovered from the schema's field
    /// names. Files that pick a different menu round-trip through this — the menu is
    /// part of the data, not source-level.
    pub fn tracked_stats(&self) -> Vec<Stat> {
        let DType::Struct(fields, _) = self.0.as_ref().dtype() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (name, dtype) in fields.names().iter().zip(fields.fields()) {
            if let Some((stat, StatFieldRole::Value)) = parse_stat_field(name.as_ref(), &dtype) {
                out.push(stat);
            }
        }
        out
    }

    /// Read node `idx` from the columns into a typed [`StatsSet`].
    ///
    /// Returns `None` if every tracked column is null/absent at that row. Proto-encoded
    /// value-stats (`min`/`max`/`sum`) are deserialized using `array_dtype` to recover
    /// the typed `ScalarValue`; the reader iterates the schema's actual stat menu, so a
    /// file that tracks a subset just produces stats from the subset.
    ///
    /// `ctx` is reused across the per-stat `execute_scalar` calls for this row.
    pub fn read(
        &self,
        idx: usize,
        array_dtype: &DType,
        ctx: &mut ExecutionCtx,
        session: &VortexSession,
    ) -> VortexResult<Option<StatsSet>> {
        let mut field = |name: &str| -> VortexResult<crate::scalar::Scalar> {
            self.0
                .unmasked_field_by_name_opt(name)
                .ok_or_else(|| vortex_err!("StatsColumns missing field {}", name))?
                .execute_scalar(idx, ctx)
        };

        let mut set = StatsSet::default();
        for stat in self.tracked_stats() {
            self.read_stat(&mut field, stat, array_dtype, session, &mut set)?;
        }
        Ok((!set.is_empty()).then_some(set))
    }

    fn read_stat(
        &self,
        field: &mut impl FnMut(&str) -> VortexResult<crate::scalar::Scalar>,
        stat: Stat,
        array_dtype: &DType,
        session: &VortexSession,
        set: &mut StatsSet,
    ) -> VortexResult<()> {
        match stat_column_kind(stat) {
            StatColumnKind::BinaryValueWithExact => {
                if let Some(bytes) = field(stat.name())?.as_binary().value().cloned()
                    && let Some(stat_dtype) = stat.dtype(array_dtype)
                    && let Some(value) =
                        ScalarValue::from_proto_bytes(bytes.as_slice(), &stat_dtype, session)?
                {
                    let exact_field = format!("{}_exact", stat.name());
                    let exact = field(&exact_field)?.as_bool().value().unwrap_or(true);
                    let kind = if exact {
                        Precision::Exact
                    } else {
                        Precision::Inexact
                    };
                    set.set(stat, kind(value));
                }
            }
            StatColumnKind::BinaryValue => {
                if let Some(bytes) = field(stat.name())?.as_binary().value().cloned()
                    && let Some(stat_dtype) = stat.dtype(array_dtype)
                    && let Some(value) =
                        ScalarValue::from_proto_bytes(bytes.as_slice(), &stat_dtype, session)?
                {
                    set.set(stat, Precision::Exact(value));
                }
            }
            StatColumnKind::U64 => {
                if let Some(v) = field(stat.name())?.as_primitive().as_::<u64>() {
                    set.set(stat, Precision::Exact(ScalarValue::from(v)));
                }
            }
            StatColumnKind::Bool => {
                if let Some(v) = field(stat.name())?.as_bool().value() {
                    set.set(stat, Precision::Exact(ScalarValue::from(v)));
                }
            }
        }
        Ok(())
    }
}

/// One column inside a [`StatsColumnsBuilder`], plus what stat + role it represents.
enum StatColumnBuilder {
    Binary {
        stat: Stat,
        field: FieldName,
        builder: VarBinViewBuilder,
    },
    Exact {
        stat: Stat,
        field: FieldName,
        builder: BoolBuilder,
    },
    U64 {
        stat: Stat,
        field: FieldName,
        builder: PrimitiveBuilder<u64>,
    },
    Bool {
        stat: Stat,
        field: FieldName,
        builder: BoolBuilder,
    },
}

impl StatColumnBuilder {
    fn field(&self) -> &FieldName {
        match self {
            Self::Binary { field, .. }
            | Self::Exact { field, .. }
            | Self::U64 { field, .. }
            | Self::Bool { field, .. } => field,
        }
    }

    fn append_null(&mut self) {
        match self {
            Self::Binary { builder, .. } => builder.append_null(),
            Self::Exact { builder, .. } | Self::Bool { builder, .. } => builder.append_null(),
            Self::U64 { builder, .. } => builder.append_null(),
        }
    }

    fn append_from(&mut self, stats: &StatsSet) {
        let bool_dtype = DType::Bool(Nullability::NonNullable);
        let u64_dtype: DType = PType::U64.into();
        match self {
            Self::Binary { stat, builder, .. } => match stat_column_kind(*stat) {
                StatColumnKind::BinaryValueWithExact => {
                    let p = stats.get(*stat);
                    match p.as_ref().into_inner() {
                        Some(v) => {
                            let bytes = ScalarValue::to_proto_bytes::<Vec<u8>>(Some(v));
                            builder.append_value(&bytes);
                        }
                        None => builder.append_null(),
                    }
                }
                StatColumnKind::BinaryValue => {
                    // exact-only by construction (e.g. Sum); inexact values are dropped.
                    match stats.get(*stat).as_exact() {
                        Some(v) => {
                            let bytes = ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&v));
                            builder.append_value(&bytes);
                        }
                        None => builder.append_null(),
                    }
                }
                _ => builder.append_null(),
            },
            Self::Exact { stat, builder, .. } => {
                let p = stats.get(*stat);
                match p.as_ref().into_inner() {
                    Some(_) => builder.append_value(p.is_exact()),
                    None => builder.append_null(),
                }
            }
            Self::U64 { stat, builder, .. } => {
                push_opt_u64(builder, stats.get_as::<u64>(*stat, &u64_dtype).as_exact())
            }
            Self::Bool { stat, builder, .. } => {
                push_opt_bool(builder, stats.get_as::<bool>(*stat, &bool_dtype).as_exact())
            }
        }
    }

    fn finish(self) -> ArrayRef {
        match self {
            Self::Binary { mut builder, .. } => builder.finish_into_varbinview().into_array(),
            Self::Exact { mut builder, .. } | Self::Bool { mut builder, .. } => {
                builder.finish_into_bool().into_array()
            }
            Self::U64 { mut builder, .. } => builder.finish_into_primitive().into_array(),
        }
    }
}

/// Streaming accumulator for [`StatsColumns`]. Push one [`StatsSet`] per node (or `None`
/// for "no stats persisted"); call [`Self::finish`] to materialize a [`StructArray`]
/// whose schema matches [`stats_columns_dtype`] over the chosen stat menu.
pub struct StatsColumnsBuilder {
    n: usize,
    columns: Vec<StatColumnBuilder>,
}

impl StatsColumnsBuilder {
    /// Build a stats accumulator tracking the given menu of stats with `capacity` rows
    /// reserved per column.
    pub fn new(stats: &[Stat], capacity: usize) -> Self {
        let nullable = Nullability::Nullable;
        let mut columns = Vec::with_capacity(stats.len() * 2);
        for &stat in stats {
            match stat_column_kind(stat) {
                StatColumnKind::BinaryValueWithExact => {
                    columns.push(StatColumnBuilder::Binary {
                        stat,
                        field: stat.name().into(),
                        builder: VarBinViewBuilder::with_capacity(
                            DType::Binary(nullable),
                            capacity,
                        ),
                    });
                    columns.push(StatColumnBuilder::Exact {
                        stat,
                        field: format!("{}_exact", stat.name()).into(),
                        builder: BoolBuilder::with_capacity(nullable, capacity),
                    });
                }
                StatColumnKind::BinaryValue => {
                    columns.push(StatColumnBuilder::Binary {
                        stat,
                        field: stat.name().into(),
                        builder: VarBinViewBuilder::with_capacity(
                            DType::Binary(nullable),
                            capacity,
                        ),
                    });
                }
                StatColumnKind::U64 => {
                    columns.push(StatColumnBuilder::U64 {
                        stat,
                        field: stat.name().into(),
                        builder: PrimitiveBuilder::<u64>::with_capacity(nullable, capacity),
                    });
                }
                StatColumnKind::Bool => {
                    columns.push(StatColumnBuilder::Bool {
                        stat,
                        field: stat.name().into(),
                        builder: BoolBuilder::with_capacity(nullable, capacity),
                    });
                }
            }
        }
        Self { n: 0, columns }
    }

    /// Convenience: build with [`DEFAULT_STATS`] as the menu.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(DEFAULT_STATS, capacity)
    }

    /// Append one node's stats. `None` writes all columns as null at this row.
    pub fn push(&mut self, stats: Option<&StatsSet>) {
        self.n += 1;
        match stats {
            Some(stats) => {
                for column in &mut self.columns {
                    column.append_from(stats);
                }
            }
            None => {
                for column in &mut self.columns {
                    column.append_null();
                }
            }
        }
    }

    /// Materialize the accumulated rows into a [`StatsColumns`].
    pub fn finish(self) -> VortexResult<StatsColumns> {
        let n = self.n;
        let names: Vec<FieldName> = self.columns.iter().map(|c| c.field().clone()).collect();
        let fields: Vec<ArrayRef> = self.columns.into_iter().map(|c| c.finish()).collect();
        let inner = StructArray::try_new(names.into(), fields, n, Validity::NonNullable)?;
        StatsColumns::new(inner)
    }
}

fn push_opt_u64(builder: &mut PrimitiveBuilder<u64>, v: Option<u64>) {
    match v {
        Some(v) => builder.append_value(v),
        None => builder.append_null(),
    }
}

fn push_opt_bool(builder: &mut BoolBuilder, v: Option<bool>) {
    match v {
        Some(v) => builder.append_value(v),
        None => builder.append_null(),
    }
}

/// Build the schema of the `nodes` struct that backs a [`ColumnarArrayTree`] (one row
/// per `ArrayNode` in pre-order) for the given stat menu.
///
/// Carries everything a `ColumnarSerializedArray` needs to navigate and decode a node:
/// encoding id, child count, plugin-specific metadata, buffer count, the precomputed
/// `subtree_size` and `buffer_offset` nav values, and a nested `stats` struct (see
/// [`stats_columns_dtype`]).
pub fn nodes_columns_dtype(stats: &[Stat]) -> DType {
    let prim = |p: PType| DType::Primitive(p, Nullability::NonNullable);
    DType::Struct(
        StructFields::new(
            NODE_FIELDS
                .iter()
                .map(|n| FieldName::from(*n))
                .collect::<Vec<_>>()
                .into(),
            vec![
                prim(PType::U16),                        // encoding_id
                prim(PType::U8),                         // child_count
                DType::Binary(Nullability::NonNullable), // metadata
                prim(PType::U16),                        // buffers_per_node
                prim(PType::U32),                        // subtree_size
                prim(PType::U32),                        // buffer_offset
                stats_columns_dtype(stats),              // stats
            ],
        ),
        Nullability::NonNullable,
    )
}

const NODE_FIELDS: [&str; 7] = [
    "encoding_id",
    "child_count",
    "metadata",
    "buffers_per_node",
    "subtree_size",
    "buffer_offset",
    "stats",
];

/// Canonical schema of the `buffers` struct that backs a [`ColumnarArrayTree`]
/// (one row per buffer descriptor, concatenated across all nodes in pre-order).
pub static BUFFER_COLUMNS_DTYPE: LazyLock<DType> = LazyLock::new(|| {
    let nn = Nullability::NonNullable;
    let prim = |p: PType| DType::Primitive(p, nn);
    DType::Struct(
        StructFields::new(
            BUFFER_FIELDS
                .iter()
                .map(|n| FieldName::from(*n))
                .collect::<Vec<_>>()
                .into(),
            vec![prim(PType::U16), prim(PType::U8), prim(PType::U32)],
        ),
        nn,
    )
});

const BUFFER_FIELDS: [&str; 3] = ["padding", "alignment_exponent", "length"];

/// Columnar representation of one `ArrayNode` tree, shared by all
/// `ColumnarSerializedArray` nodes that navigate it via `Arc`.
///
/// The private typed-field handles below are zero-copy [`Arc`] clones of the underlying
/// struct fields, supplied to [`Self::try_new`] so per-node decode access doesn't
/// pay a field-name lookup + downcast per call.
///
/// `subtree_size` and `buffer_offset` give O(1) child navigation / buffer slicing.
#[derive(Debug, Clone)]
pub struct ColumnarArrayTree {
    /// Canonical `NODES_COLUMNS_DTYPE` struct, one row per array node.
    pub nodes: StructArray,
    /// Canonical `BUFFERS_COLUMNS_DTYPE` struct, one row per buffer descriptor.
    pub buffers: StructArray,

    // Cached references to the columns of the struct arrays above
    encoding_ids: PrimitiveArray,
    child_counts: PrimitiveArray,
    node_metadata: VarBinViewArray,
    buffers_per_node: PrimitiveArray,
    subtree_sizes: PrimitiveArray,
    buffer_offsets: PrimitiveArray,
    buffer_padding: PrimitiveArray,
    buffer_alignment_exponent: PrimitiveArray,
    buffer_length: PrimitiveArray,
    stats: StatsColumns,
}

/// Compute `subtree_sizes` from `child_counts` via a single right-to-left pass.
///
/// In pre-order traversal, a node's subtree occupies a contiguous range of indices
/// starting at the node, so a node's subtree size is `1 + sum of its children's subtree
/// sizes`. Iterating right-to-left guarantees each node's children have already been
/// visited when we reach it.
pub fn compute_subtree_sizes(child_counts: &[u8]) -> Buffer<u32> {
    let n = child_counts.len();
    let mut sizes = vec![0u32; n];
    for i in (0..n).rev() {
        let mut total = 1u32;
        let mut cursor = i + 1;
        for _ in 0..child_counts[i] {
            let child_size = sizes[cursor];
            total += child_size;
            cursor += child_size as usize;
        }
        sizes[i] = total;
    }
    Buffer::from(sizes)
}

/// Compute `buffer_offsets` as a prefix sum of `buffers_per_node`.
pub fn compute_buffer_offsets(buffers_per_node: &[u16]) -> Buffer<u32> {
    buffers_per_node
        .iter()
        .scan(0u32, |acc, &n| {
            let out = *acc;
            *acc += n as u32;
            Some(out)
        })
        .collect::<Vec<_>>()
        .into()
}

impl ColumnarArrayTree {
    /// Construct a `ColumnarArrayTree` from typed per-node and per-buffer columns,
    /// assembling the canonical `nodes` and `buffers` [`StructArray`]s in the process.
    ///
    /// The input types — `PrimitiveArray` / `VarBinViewArray` / [`StatsColumns`] — match
    /// the outer slots of [`NODES_COLUMNS_DTYPE`] and [`BUFFER_COLUMNS_DTYPE`], so dtype
    /// validation collapses to length consistency, which [`StructArray::try_new`] checks.
    /// `PrimitiveArray::ptype` per column is trusted: writer callers always go through
    /// the typed helpers, and reader callers must downcast field-by-field before calling.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        encoding_ids: PrimitiveArray,
        child_counts: PrimitiveArray,
        node_metadata: VarBinViewArray,
        buffers_per_node: PrimitiveArray,
        subtree_sizes: PrimitiveArray,
        buffer_offsets: PrimitiveArray,
        stats: StatsColumns,
        buffer_padding: PrimitiveArray,
        buffer_alignment_exponent: PrimitiveArray,
        buffer_length: PrimitiveArray,
    ) -> VortexResult<Self> {
        let n_nodes = encoding_ids.as_ref().len();
        let n_buffers = buffer_padding.as_ref().len();

        let nodes = StructArray::try_new(
            NODE_FIELDS
                .iter()
                .map(|s| FieldName::from(*s))
                .collect::<Vec<_>>()
                .into(),
            vec![
                encoding_ids.clone().into_array(),
                child_counts.clone().into_array(),
                node_metadata.clone().into_array(),
                buffers_per_node.clone().into_array(),
                subtree_sizes.clone().into_array(),
                buffer_offsets.clone().into_array(),
                stats.clone().into_struct().into_array(),
            ],
            n_nodes,
            Validity::NonNullable,
        )?;

        let buffers = StructArray::try_new(
            BUFFER_FIELDS
                .iter()
                .map(|s| FieldName::from(*s))
                .collect::<Vec<_>>()
                .into(),
            vec![
                buffer_padding.clone().into_array(),
                buffer_alignment_exponent.clone().into_array(),
                buffer_length.clone().into_array(),
            ],
            n_buffers,
            Validity::NonNullable,
        )?;

        Ok(Self {
            nodes,
            buffers,
            encoding_ids,
            child_counts,
            node_metadata,
            buffers_per_node,
            subtree_sizes,
            buffer_offsets,
            buffer_padding,
            buffer_alignment_exponent,
            buffer_length,
            stats,
        })
    }

    /// Number of nodes in the tree.
    pub fn nnodes(&self) -> usize {
        self.nodes.as_ref().len()
    }
}

/// Parallel to [`crate::serde::SerializedArray`] but sourced from a columnar representation
/// of the encoding tree rather than a flatbuffer.
///
/// Holds an `Arc<ColumnarArrayTree>` plus a `node_index` that identifies the current
/// node within the tree. `child(idx)` returns a new `ColumnarSerializedArray` pointing
/// at the requested child by computing the child's pre-order index from
/// `subtree_sizes`.
///
/// `decode()` performs the same plugin dispatch as `SerializedArray::decode`, just sourcing
/// metadata/buffers/stats from the columnar tree.
#[derive(Clone)]
pub struct ColumnarSerializedArray {
    tree: Arc<ColumnarArrayTree>,
    node_index: usize,
    buffers: Arc<[BufferHandle]>,
}

impl Debug for ColumnarSerializedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnarSerializedArray")
            .field("encoding_id", &self.encoding_id())
            .field("node_index", &self.node_index)
            .field("nchildren", &self.nchildren())
            .field("nbuffers", &self.nbuffers())
            .finish()
    }
}

impl ColumnarSerializedArray {
    /// Construct a new root-level `ColumnarSerializedArray` for the given tree.
    pub fn new(tree: Arc<ColumnarArrayTree>, buffers: Arc<[BufferHandle]>) -> VortexResult<Self> {
        if tree.nnodes() == 0 {
            vortex_bail!("ColumnarArrayTree must have at least one node");
        }
        Ok(Self {
            tree,
            node_index: 0,
            buffers,
        })
    }

    /// Slice the data segment into per-buffer handles using the descriptors in `tree`,
    /// then construct a root-level `ColumnarSerializedArray`.
    ///
    /// The segment is expected to be data-only — no trailing flatbuffer or length suffix —
    /// as produced by [`serialize_to_columnar_tree`].
    pub fn from_segment_and_tree(
        segment: BufferHandle,
        tree: Arc<ColumnarArrayTree>,
    ) -> VortexResult<Self> {
        let segment = segment.ensure_aligned(Alignment::none())?;
        let n_buffers = tree.buffer_length.as_ref().len();
        let padding = tree.buffer_padding.as_slice::<u16>();
        let lengths = tree.buffer_length.as_slice::<u32>();
        let alignments = tree.buffer_alignment_exponent.as_slice::<u8>();
        let mut handles: Vec<BufferHandle> = Vec::with_capacity(n_buffers);
        let mut offset = 0;
        for i in 0..n_buffers {
            offset += padding[i] as usize;
            let buffer_len = lengths[i] as usize;
            let alignment = Alignment::from_exponent(alignments[i]);
            let buffer = segment.slice(offset..(offset + buffer_len));
            handles.push(buffer.ensure_aligned(alignment)?);
            offset += buffer_len;
        }
        Self::new(tree, Arc::from(handles))
    }

    /// Returns the encoding id (as the interned `u16` in the file's `ArrayContext`) of the
    /// current node.
    pub fn encoding_id(&self) -> u16 {
        self.tree.encoding_ids.as_slice::<u16>()[self.node_index]
    }

    /// Returns the metadata bytes for the current node.
    pub fn metadata(&self) -> ByteBuffer {
        self.tree.node_metadata.bytes_at(self.node_index)
    }

    /// Returns the number of direct children of the current node.
    pub fn nchildren(&self) -> usize {
        self.tree.child_counts.as_slice::<u8>()[self.node_index] as usize
    }

    /// Returns a `ColumnarSerializedArray` pointing at the `idx`th direct child of the
    /// current node.
    pub fn child(&self, idx: usize) -> ColumnarSerializedArray {
        let n_children = self.nchildren();
        if idx >= n_children {
            vortex_panic!(
                "Invalid child index {} for node with {} children",
                idx,
                n_children
            );
        }
        // Children are laid out in pre-order immediately after the current node. The first
        // child is at node_index + 1; each subsequent child sits at the previous child's
        // index + that child's subtree size.
        let subtree_sizes = self.tree.subtree_sizes.as_slice::<u32>();
        let mut cursor = self.node_index + 1;
        for _ in 0..idx {
            cursor += subtree_sizes[cursor] as usize;
        }
        Self {
            tree: Arc::clone(&self.tree),
            node_index: cursor,
            buffers: Arc::clone(&self.buffers),
        }
    }

    /// Number of buffers owned by the current node.
    pub fn nbuffers(&self) -> usize {
        self.tree.buffers_per_node.as_slice::<u16>()[self.node_index] as usize
    }

    /// Return the slice of buffer handles owned by the current node.
    fn node_buffers(&self) -> VortexResult<&[BufferHandle]> {
        let start = self.tree.buffer_offsets.as_slice::<u32>()[self.node_index] as usize;
        let count = self.nbuffers();
        self.buffers.get(start..start + count).ok_or_else(|| {
            vortex_err!(
                "buffer indices {}..{} out of range for {} buffers",
                start,
                start + count,
                self.buffers.len(),
            )
        })
    }

    /// Decode this node into an `ArrayRef` using the same plugin contract as
    /// [`crate::serde::SerializedArray::decode`].
    pub fn decode(
        &self,
        dtype: &DType,
        len: usize,
        ctx: &ReadContext,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let encoding_idx = self.encoding_id();
        let encoding_id = ctx
            .resolve(encoding_idx)
            .ok_or_else(|| vortex_err!("Unknown encoding index: {}", encoding_idx))?;
        let plugin = session
            .arrays()
            .registry()
            .find(&encoding_id)
            .ok_or_else(|| vortex_err!("Unknown encoding: {}", encoding_id))?;

        let buffers = self.node_buffers()?;
        let children = ColumnarSerializedArrayChildren {
            ser: self,
            ctx,
            session,
        };

        let metadata = self.metadata();
        let decoded =
            plugin.deserialize(dtype, len, metadata.as_slice(), buffers, &children, session)?;

        assert_eq!(
            decoded.len(),
            len,
            "Array decoded from {} has incorrect length {}, expected {}",
            encoding_id,
            decoded.len(),
            len
        );
        assert_eq!(
            decoded.dtype(),
            dtype,
            "Array decoded from {} has incorrect dtype {}, expected {}",
            encoding_id,
            decoded.dtype(),
            dtype,
        );
        assert!(
            plugin.is_supported_encoding(&decoded.encoding_id()),
            "Array decoded from {} has incorrect encoding {}",
            encoding_id,
            decoded.encoding_id(),
        );

        // Populate statistics from the columnar tree. `StatsColumns::read` walks
        // the 11 stat columns at this node's row and rehydrates a `StatsSet`, decoding
        // min/max/sum proto bytes using the now-known dtype. We create a temporary
        // `ExecutionCtx` per node decode — see the discussion in the columnar module
        // header about why we don't thread one through the recursive children machinery.
        let mut stats_ctx = session.create_execution_ctx();
        if let Some(stats_set) =
            self.tree
                .stats
                .read(self.node_index, dtype, &mut stats_ctx, session)?
        {
            decoded.statistics().set_iter(stats_set.into_iter());
        }

        Ok(decoded)
    }
}

struct ColumnarSerializedArrayChildren<'a> {
    ser: &'a ColumnarSerializedArray,
    ctx: &'a ReadContext,
    session: &'a VortexSession,
}

impl ArrayChildren for ColumnarSerializedArrayChildren<'_> {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        self.ser
            .child(index)
            .decode(dtype, len, self.ctx, self.session)
    }

    fn len(&self) -> usize {
        self.ser.nchildren()
    }
}

/// Writer-side entry point. Walks `array` in pre-order once and produces:
///
/// 1. The data-segment buffer list (data buffers only, no trailing flatbuffer or length
///    suffix — segments are not self-contained and must be paired with the
///    [`ColumnarArrayTree`] to decode).
/// 2. A [`ColumnarArrayTree`] capturing the encoding tree, per-node stats, per-buffer
///    descriptors, and the precomputed `subtree_sizes` / `buffer_offsets` nav columns.
pub fn serialize_to_columnar_tree(
    array: &ArrayRef,
    ctx: &ArrayContext,
    session: &VortexSession,
    options: &SerializeOptions,
) -> VortexResult<(Vec<ByteBuffer>, ColumnarArrayTree)> {
    // Per-node columns collected during the DFS walk.
    let mut encoding_ids = Vec::new();
    let mut child_counts = Vec::new();
    let mut node_metadata = Vec::new();
    let mut buffers_per_node = Vec::new();
    let mut stats_builder = StatsColumnsBuilder::with_capacity(0);
    // Flat list of all data buffers across all nodes, in pre-order.
    let mut array_buffers = Vec::new();

    for node in array.depth_first_traversal() {
        let encoding_idx = ctx.intern(&node.encoding_id()).ok_or_else(|| {
            vortex_err!("Array encoding {} not permitted by ctx", node.encoding_id())
        })?;
        encoding_ids.push(encoding_idx);

        let n_children = u8::try_from(node.nchildren())
            .map_err(|_| vortex_err!("Array node has more than u8::MAX children"))?;
        child_counts.push(n_children);

        let metadata_bytes = session.array_serialize(&node)?.ok_or_else(|| {
            vortex_err!(
                "Array {} does not support serialization",
                node.encoding_id()
            )
        })?;
        node_metadata.push(ByteBuffer::from(metadata_bytes));

        let node_bufs = node.buffers();
        let n_buffers = u16::try_from(node_bufs.len())
            .map_err(|_| vortex_err!("Array node has more than u16::MAX buffers"))?;
        buffers_per_node.push(n_buffers);

        // Snapshot the current StatsSet straight into the per-stat columns. Empty sets
        // push all-nulls — semantically identical to "no stats persisted" since the read
        // side treats all-null as `None` from `StatsColumns::read`.
        let stats_set = node.statistics().to_owned();
        stats_builder.push(if stats_set.is_empty() {
            None
        } else {
            Some(&stats_set)
        });

        array_buffers.extend(node_bufs);
    }

    // Emit the data buffer list and per-buffer descriptor columns in one pass. Padding
    // math is the same rule the inline flatbuffer path uses: each buffer is padded to
    // its required alignment, tracked through a running `pos` cursor.
    let max_alignment = array_buffers
        .iter()
        .map(|buf| buf.alignment())
        .max()
        .unwrap_or(Alignment::none());
    let zeros = ByteBuffer::zeroed(*max_alignment);

    let mut buffers = vec![ByteBuffer::zeroed_aligned(0, max_alignment)];
    let mut buffer_padding = Vec::<u16>::with_capacity(array_buffers.len());
    let mut buffer_alignment_exponent = Vec::<u8>::with_capacity(array_buffers.len());
    let mut buffer_length = Vec::<u32>::with_capacity(array_buffers.len());
    let mut pos = options.offset;

    for buffer in &array_buffers {
        let padding = if options.include_padding {
            let padding = pos.next_multiple_of(*buffer.alignment()) - pos;
            if padding > 0 {
                pos += padding;
                buffers.push(zeros.slice(0..padding));
            }
            padding
        } else {
            0
        };
        buffer_padding
            .push(u16::try_from(padding).map_err(|_| vortex_err!("buffer padding overflows u16"))?);
        buffer_alignment_exponent.push(buffer.alignment().exponent());
        buffer_length.push(
            u32::try_from(buffer.len()).map_err(|_| vortex_err!("buffer length overflows u32"))?,
        );

        pos += buffer.len();
        buffers.push(buffer.clone().aligned(Alignment::none()));
    }

    // these two precomputed columns help O(1) child access on read
    let subtree_sizes = compute_subtree_sizes(&child_counts);
    let buffer_offsets = compute_buffer_offsets(&buffers_per_node);

    let node_metadata = VarBinViewArray::from_iter_bin(node_metadata.iter().map(|b| b.as_slice()));
    let stats = stats_builder.finish()?;

    let tree = ColumnarArrayTree::try_new(
        primitive_array_u16(encoding_ids),
        primitive_array_u8(child_counts),
        node_metadata,
        primitive_array_u16(buffers_per_node),
        primitive_array_u32_buffer(subtree_sizes),
        primitive_array_u32_buffer(buffer_offsets),
        stats,
        primitive_array_u16(buffer_padding),
        primitive_array_u8(buffer_alignment_exponent),
        primitive_array_u32(buffer_length),
    )?;

    Ok((buffers, tree))
}

fn primitive_array_u16(v: Vec<u16>) -> PrimitiveArray {
    PrimitiveArray::new(Buffer::from(v), Validity::NonNullable)
}
fn primitive_array_u8(v: Vec<u8>) -> PrimitiveArray {
    PrimitiveArray::new(Buffer::from(v), Validity::NonNullable)
}
fn primitive_array_u32(v: Vec<u32>) -> PrimitiveArray {
    PrimitiveArray::new(Buffer::from(v), Validity::NonNullable)
}
fn primitive_array_u32_buffer(b: Buffer<u32>) -> PrimitiveArray {
    PrimitiveArray::new(b, Validity::NonNullable)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    /// Tree shape:
    ///   0 (root, 2 children)
    ///   ├── 1 (leaf)
    ///   └── 2 (1 child)
    ///       └── 3 (leaf)
    /// Subtree sizes: [4, 1, 2, 1].
    #[test]
    fn subtree_sizes_basic() -> VortexResult<()> {
        let child_counts = vec![2u8, 0, 1, 0];
        let sizes = compute_subtree_sizes(&child_counts);
        assert_eq!(sizes.as_slice(), &[4, 1, 2, 1]);
        Ok(())
    }

    /// Single-node tree.
    #[test]
    fn subtree_sizes_leaf() -> VortexResult<()> {
        let sizes = compute_subtree_sizes(&[0u8]);
        assert_eq!(sizes.as_slice(), &[1]);
        Ok(())
    }

    /// Deeply nested tree (left-skewed):
    ///   0 -> 1 -> 2 -> 3 (leaf)
    /// Subtree sizes: [4, 3, 2, 1].
    #[test]
    fn subtree_sizes_skewed() -> VortexResult<()> {
        let sizes = compute_subtree_sizes(&[1u8, 1, 1, 0]);
        assert_eq!(sizes.as_slice(), &[4, 3, 2, 1]);
        Ok(())
    }

    #[test]
    fn buffer_offsets_basic() {
        let offsets = compute_buffer_offsets(&[2u16, 0, 3, 1]);
        assert_eq!(offsets.as_slice(), &[0, 2, 2, 5]);
    }

    /// Round-trip a populated `StatsSet` through the `StatsColumnsBuilder` ->
    /// `StatsColumns::read` path to confirm the per-stat columns preserve the same
    /// selection of stats and their values across the columnar wire format.
    #[test]
    fn stats_columns_roundtrip_i32() -> VortexResult<()> {
        use crate::LEGACY_SESSION;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut set = StatsSet::default();
        set.set(Stat::Min, Precision::Exact(ScalarValue::from(-3i32)));
        set.set(Stat::Max, Precision::Inexact(ScalarValue::from(42i32)));
        set.set(Stat::Sum, Precision::Exact(ScalarValue::from(100i64)));
        set.set(Stat::NullCount, Precision::Exact(ScalarValue::from(7u64)));
        set.set(Stat::IsConstant, Precision::Exact(ScalarValue::from(false)));
        set.set(Stat::IsSorted, Precision::Exact(ScalarValue::from(true)));

        let mut builder = StatsColumnsBuilder::with_capacity(1);
        builder.push(Some(&set));
        let cols = builder.finish()?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let back = cols
            .read(0, &dtype, &mut ctx, &LEGACY_SESSION)?
            .expect("non-empty");
        assert_eq!(back.get_as::<i32>(Stat::Min, &dtype), Precision::Exact(-3));
        assert_eq!(
            back.get_as::<i32>(Stat::Max, &dtype),
            Precision::Inexact(42)
        );
        assert_eq!(
            back.get_as::<u64>(Stat::NullCount, &PType::U64.into()),
            Precision::Exact(7)
        );
        assert_eq!(
            back.get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable)),
            Precision::Exact(false)
        );
        assert_eq!(
            back.get_as::<bool>(Stat::IsSorted, &DType::Bool(Nullability::NonNullable)),
            Precision::Exact(true)
        );
        assert!(
            back.get(Stat::IsStrictSorted).is_absent(),
            "unset stats stay unset"
        );
        Ok(())
    }

    /// Empty stats round-trip to `None` (every column at the row is null).
    #[test]
    fn stats_columns_empty() -> VortexResult<()> {
        use crate::LEGACY_SESSION;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut builder = StatsColumnsBuilder::with_capacity(1);
        builder.push(None);
        let cols = builder.finish()?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(cols.read(0, &dtype, &mut ctx, &LEGACY_SESSION)?.is_none());
        Ok(())
    }

    /// Child navigation: from root (idx 0) of a tree
    ///   0 [2 children]
    ///   ├── 1 [leaf]
    ///   └── 2 [1 child]
    ///       └── 3 [leaf]
    /// expect child(0) -> node 1, child(1) -> node 2. Then from node 2, child(0) -> node 3.
    #[test]
    fn child_navigation() -> VortexResult<()> {
        let child_counts = vec![2u8, 0, 1, 0];
        let buffers_per_node = vec![0u16; 4];
        let subtree_sizes = compute_subtree_sizes(&child_counts);
        let buffer_offsets = compute_buffer_offsets(&buffers_per_node);

        let stats = {
            let mut b = StatsColumnsBuilder::with_capacity(4);
            for _ in 0..4 {
                b.push(None);
            }
            b.finish()?
        };

        let tree = Arc::new(ColumnarArrayTree::try_new(
            primitive_array_u16(vec![0u16, 1, 2, 3]),
            primitive_array_u8(child_counts),
            VarBinViewArray::from_iter_bin(iter::repeat_n(b"".as_slice(), 4)),
            primitive_array_u16(buffers_per_node),
            primitive_array_u32_buffer(subtree_sizes),
            primitive_array_u32_buffer(buffer_offsets),
            stats,
            primitive_array_u16(Vec::new()),
            primitive_array_u8(Vec::new()),
            primitive_array_u32(Vec::new()),
        )?);
        let root = ColumnarSerializedArray::new(tree, Arc::new([]))?;
        assert_eq!(root.encoding_id(), 0);
        assert_eq!(root.nchildren(), 2);
        let c0 = root.child(0);
        assert_eq!(c0.encoding_id(), 1);
        assert_eq!(c0.nchildren(), 0);
        let c1 = root.child(1);
        assert_eq!(c1.encoding_id(), 2);
        assert_eq!(c1.nchildren(), 1);
        let c1c0 = c1.child(0);
        assert_eq!(c1c0.encoding_id(), 3);
        assert_eq!(c1c0.nchildren(), 0);
        Ok(())
    }
}
