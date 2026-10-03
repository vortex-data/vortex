// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::decode::decode_to_map;
use crate::rules::RULES;

/// A [`ShreddedMap`]-encoded Vortex array.
pub type ShreddedMapArray = Array<ShreddedMap>;

/// A map encoding that stores frequently occurring keys in dedicated columns.
///
/// The logical dtype is a [`DType::Map`] with non-nullable UTF-8 keys and `keys_sorted = true`.
/// Each shredded key owns one nullable column that is row-aligned with the map. A non-null value
/// at row `i` means the map at row `i` contains `(key, value)`. Every other entry lives in the
/// `residual` child, a map of the same dtype that also carries the outer validity.
///
/// A row's entries decode as the sorted merge of its present shredded keys and its residual
/// entries, with the shredded entry first on equal keys. The shredder therefore only moves the
/// first entry of each key in a row, and only when its value is non-null, which keeps decoding
/// lossless for duplicate keys and null values.
///
/// For a union value dtype, a column whose values all select the same variant stores that
/// variant's child directly ("typed" shredding) instead of a sparse union.
#[derive(Clone, Debug)]
pub struct ShreddedMap;

#[array_slots(ShreddedMap)]
pub struct ShreddedMapSlots {
    /// Map of every entry that was not shredded, carrying the outer validity.
    #[slot(0)]
    pub residual: ArrayRef,
    /// One row-aligned nullable column per shredded key, in key order.
    #[slot(1..)]
    pub columns: Vec<ArrayRef>,
}

/// Describes one shredded column.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShreddedColumn {
    /// The map key stored in this column.
    pub key: Arc<str>,
    /// For a union value dtype, the child index of the single variant the column stores. `None`
    /// means the column stores the full value dtype.
    pub variant: Option<usize>,
}

/// Encoding metadata of a [`ShreddedMapArray`]: the shredded columns, sorted by key.
#[derive(Clone, Debug, Default)]
pub struct ShreddedMapData {
    columns: Arc<[ShreddedColumn]>,
}

impl Display for ShreddedMapData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "columns: [")?;
        for (i, column) in self.columns.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", column.key)?;
            if let Some(variant) = column.variant {
                write!(f, "@{variant}")?;
            }
        }
        write!(f, "]")
    }
}

impl ArrayEq for ShreddedMapData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.columns == other.columns
    }
}

impl ArrayHash for ShreddedMapData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.columns.hash(state);
    }
}

impl ShreddedMapData {
    /// The shredded columns, sorted by key.
    pub fn columns(&self) -> &[ShreddedColumn] {
        &self.columns
    }
}

#[derive(Clone, prost::Message)]
struct ShreddedColumnProto {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(uint32, optional, tag = "2")]
    variant: Option<u32>,
}

#[derive(Clone, prost::Message)]
struct ShreddedMapMetadataProto {
    #[prost(message, repeated, tag = "1")]
    columns: Vec<ShreddedColumnProto>,
}

/// Returns the dtype a shredded column must have.
pub(crate) fn column_dtype(map_dtype: &MapDType, column: &ShreddedColumn) -> VortexResult<DType> {
    let value_dtype = map_dtype.value_dtype();
    let dtype = match column.variant {
        None => value_dtype,
        Some(variant) => {
            let DType::Union(variants, _) = &value_dtype else {
                vortex_bail!("typed shredded column requires a union value dtype, got {value_dtype}");
            };
            variants
                .variant_by_index(variant)
                .ok_or_else(|| vortex_err!("variant index {variant} out of bounds"))?
        }
    };
    Ok(dtype.as_nullable())
}

fn validate_parts(
    dtype: &DType,
    len: usize,
    columns: &[ShreddedColumn],
    slots: &[Option<ArrayRef>],
) -> VortexResult<()> {
    let DType::Map(map_dtype, _) = dtype else {
        vortex_bail!("ShreddedMap requires a map dtype, got {dtype}");
    };
    vortex_ensure!(
        matches!(map_dtype.key_dtype(), DType::Utf8(_)),
        "ShreddedMap requires UTF-8 keys, got {}",
        map_dtype.key_dtype()
    );
    vortex_ensure!(
        map_dtype.keys_sorted(),
        "ShreddedMap requires keys_sorted maps so decoding can merge entries in order"
    );
    vortex_ensure!(
        slots.len() == ShreddedMapSlots::COLUMNS_OFFSET + columns.len(),
        "ShreddedMap expected {} slots, found {}",
        ShreddedMapSlots::COLUMNS_OFFSET + columns.len(),
        slots.len()
    );
    vortex_ensure!(
        columns.windows(2).all(|w| w[0].key < w[1].key),
        "ShreddedMap column keys must be strictly sorted"
    );

    let view = ShreddedMapSlotsView::from_slots(slots);
    vortex_ensure!(
        view.residual.dtype() == dtype,
        "residual dtype {} does not match {dtype}",
        view.residual.dtype()
    );
    vortex_ensure!(view.residual.len() == len, "residual length mismatch");
    for (column, array) in columns.iter().zip(view.columns.iter()) {
        let expected = column_dtype(map_dtype, column)?;
        vortex_ensure!(
            array.dtype() == &expected,
            "column {} has dtype {}, expected {expected}",
            column.key,
            array.dtype()
        );
        vortex_ensure!(array.len() == len, "column {} length mismatch", column.key);
    }
    Ok(())
}

pub(crate) fn make_parts(
    dtype: DType,
    len: usize,
    columns: Arc<[ShreddedColumn]>,
    residual: ArrayRef,
    column_arrays: impl IntoIterator<Item = ArrayRef>,
) -> ArrayParts<ShreddedMap> {
    let mut slots = ArraySlots::with_capacity(ShreddedMapSlots::COLUMNS_OFFSET + columns.len());
    slots.push(Some(residual));
    slots.extend(column_arrays.into_iter().map(Some));
    ArrayParts::new(ShreddedMap, dtype, len, ShreddedMapData { columns }).with_slots(slots)
}

impl ShreddedMap {
    /// Constructs a shredded map from its residual map and shredded columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the parts do not match the layout described on [`ShreddedMap`]. The
    /// data-level invariants (no residual entry that should have been shredded) are not checked.
    pub fn try_new(
        residual: ArrayRef,
        columns: Vec<ShreddedColumn>,
        column_arrays: Vec<ArrayRef>,
    ) -> VortexResult<ShreddedMapArray> {
        let dtype = residual.dtype().clone();
        let len = residual.len();
        vortex_ensure!(
            columns.len() == column_arrays.len(),
            "column metadata and arrays differ in length"
        );
        Array::try_from_parts(make_parts(
            dtype,
            len,
            columns.into(),
            residual,
            column_arrays,
        ))
    }
}

/// Compresses every child of a shredded map with `compress`, keeping dictionary columns as
/// dictionaries by compressing their codes and values separately.
///
/// Generic compressors canonicalize their input first, which would decode the shredded layout
/// back into a map and every dictionary column into one value per row.
///
/// # Errors
///
/// Returns an error if `compress` fails or changes a child's dtype or length.
pub fn compress_shredded(
    array: &ShreddedMapArray,
    mut compress: impl FnMut(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<ShreddedMapArray> {
    let residual = compress(array.residual())?;
    let columns = array
        .columns()
        .iter()
        .map(|column| compress_column(column, &mut compress))
        .collect::<VortexResult<Vec<_>>>()?;
    ShreddedMap::try_new(residual, array.data().columns().to_vec(), columns)
}

/// Compresses a shredded column through its dictionary and sparse layers.
fn compress_column(
    column: &ArrayRef,
    compress: &mut impl FnMut(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<ArrayRef> {
    if let Some(dict) = column.as_opt::<Dict>() {
        let codes = compress(dict.codes())?;
        let values = compress(dict.values())?;
        return Ok(DictArray::try_new(codes, values)?.into_array());
    }
    if column.is::<vortex_sparse::Sparse>() {
        let slots = column
            .slots()
            .iter()
            .map(|slot| slot.as_ref().map(|c| compress_column(c, compress)).transpose())
            .collect::<VortexResult<_>>()?;
        // SAFETY: compression keeps every child's dtype, length and values.
        return unsafe { column.clone().with_slots(slots) };
    }
    compress(column)
}

/// Compresses the output of [`encode`](crate::encode) child by child, see [`compress_shredded`].
///
/// # Errors
///
/// Returns an error if `compress` fails or changes a child's dtype or length.
pub fn compress_encoded(
    array: &ArrayRef,
    mut compress: impl FnMut(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<ArrayRef> {
    if let Some(dict) = array.as_opt::<Dict>()
        && let Ok(values) = dict.values().clone().try_downcast::<ShreddedMap>()
    {
        let codes = compress(dict.codes())?;
        let values = compress_shredded(&values, compress)?;
        return Ok(DictArray::try_new(codes, values.into_array())?.into_array());
    }
    match array.clone().try_downcast::<ShreddedMap>() {
        Ok(shredded) => Ok(compress_shredded(&shredded, compress)?.into_array()),
        Err(array) => compress(&array),
    }
}

/// Accessors for a [`ShreddedMapArray`].
pub trait ShreddedMapArrayExt: ShreddedMapArraySlotsExt {
    /// The map dtype of this array.
    fn map_dtype(&self) -> &MapDType {
        self.as_ref()
            .dtype()
            .as_map_opt()
            .vortex_expect("ShreddedMap requires a map dtype")
    }
}
impl<T: TypedArrayRef<ShreddedMap>> ShreddedMapArrayExt for T {}

impl VTable for ShreddedMap {
    type TypedArrayData = ShreddedMapData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.shredded_map");
        *ID
    }

    fn validate(
        &self,
        data: &ShreddedMapData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        validate_parts(dtype, len, &data.columns, slots)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ShreddedMapArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let proto = ShreddedMapMetadataProto {
            columns: array
                .data()
                .columns
                .iter()
                .map(|c| ShreddedColumnProto {
                    key: c.key.to_string(),
                    variant: c.variant.map(|v| v as u32),
                })
                .collect(),
        };
        Ok(Some(proto.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(buffers.is_empty(), "ShreddedMapArray expects no buffers");
        let DType::Map(map_dtype, _) = dtype else {
            vortex_bail!("Expected map dtype, got {dtype}");
        };
        let proto = ShreddedMapMetadataProto::decode(metadata)
            .map_err(|e| vortex_err!("invalid ShreddedMap metadata: {e}"))?;
        let columns: Arc<[ShreddedColumn]> = proto
            .columns
            .into_iter()
            .map(|c| ShreddedColumn {
                key: c.key.into(),
                variant: c.variant.map(|v| v as usize),
            })
            .collect();
        vortex_ensure!(
            children.len() == ShreddedMapSlots::COLUMNS_OFFSET + columns.len(),
            "ShreddedMapArray expected {} children, found {}",
            ShreddedMapSlots::COLUMNS_OFFSET + columns.len(),
            children.len()
        );
        let residual = children.get(ShreddedMapSlots::RESIDUAL, dtype, len)?;
        let column_arrays = columns
            .iter()
            .enumerate()
            .map(|(i, column)| {
                children.get(
                    ShreddedMapSlots::COLUMNS_OFFSET + i,
                    &column_dtype(map_dtype, column)?,
                    len,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(make_parts(
            dtype.clone(),
            len,
            columns,
            residual,
            column_arrays,
        ))
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == ShreddedMapSlots::RESIDUAL {
            "residual".to_string()
        } else {
            format!(
                "column[{}]",
                array.data().columns[idx - ShreddedMapSlots::COLUMNS_OFFSET].key
            )
        }
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decode_to_map(array.as_view(), ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<ShreddedMap> for ShreddedMap {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, ShreddedMap>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let row = array.array().slice(index..index + 1)?;
        row.execute::<MapArray>(ctx)?
            .into_array()
            .execute_scalar(0, ctx)
    }
}

impl ValidityChild<ShreddedMap> for ShreddedMap {
    fn validity_child(array: ArrayView<'_, ShreddedMap>) -> ArrayRef {
        array.residual().clone()
    }
}
