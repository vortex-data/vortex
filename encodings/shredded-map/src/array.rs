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
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
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
/// The shredded keys are the fields of one non-nullable `fields` struct, in key order. Each field
/// is a nullable column row-aligned with the map: a non-null value at row `i` means the map at row
/// `i` contains `(key, value)`. Every other entry lives in the `residual` child, a map of the same
/// dtype that also carries the outer validity. This is one node of the shredding contract: a
/// null field value means the key is absent or null in that row, never that its value lives in
/// the residual.
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
    /// Optional non-nullable booleans marking rows equal to their predecessor. A set row must
    /// equal the previous row in the residual and every column; decoding then reuses that row's
    /// entries instead of proving the equality itself.
    #[slot(1)]
    pub repeats: Option<ArrayRef>,
    /// A non-nullable struct with one row-aligned nullable field per shredded key, in key order.
    #[slot(2)]
    pub fields: ArrayRef,
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
    #[prost(bool, tag = "2")]
    has_repeats: bool,
}

/// Returns the dtype a shredded column must have.
pub(crate) fn column_dtype(map_dtype: &MapDType, column: &ShreddedColumn) -> VortexResult<DType> {
    let value_dtype = map_dtype.value_dtype();
    let dtype = match column.variant {
        None => value_dtype,
        Some(variant) => {
            let DType::Union(variants, _) = &value_dtype else {
                vortex_bail!(
                    "typed shredded column requires a union value dtype, got {value_dtype}"
                );
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
        slots.len() == ShreddedMapSlots::COUNT,
        "ShreddedMap expected {} slots, found {}",
        ShreddedMapSlots::COUNT,
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
    if let Some(repeats) = view.repeats {
        vortex_ensure!(
            repeats.dtype() == &DType::Bool(Nullability::NonNullable),
            "repeats must be non-nullable booleans, got {}",
            repeats.dtype()
        );
        vortex_ensure!(repeats.len() == len, "repeats length mismatch");
    }
    let expected = fields_dtype(map_dtype, columns)?;
    vortex_ensure!(
        view.fields.dtype() == &expected,
        "fields have dtype {}, expected {expected}",
        view.fields.dtype()
    );
    vortex_ensure!(view.fields.len() == len, "fields length mismatch");
    let fields = view
        .fields
        .as_opt::<Struct>()
        .ok_or_else(|| vortex_err!("ShreddedMap fields must be a struct array"))?;
    for (column, array) in columns.iter().zip(fields.iter_unmasked_fields()) {
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

/// The dtype of the `fields` struct: one nullable field per shredded column, named by its key.
pub(crate) fn fields_dtype(
    map_dtype: &MapDType,
    columns: &[ShreddedColumn],
) -> VortexResult<DType> {
    let dtypes = columns
        .iter()
        .map(|column| column_dtype(map_dtype, column))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(DType::Struct(
        StructFields::new(field_names(columns), dtypes),
        Nullability::NonNullable,
    ))
}

fn field_names(columns: &[ShreddedColumn]) -> FieldNames {
    FieldNames::from_iter(columns.iter().map(|c| FieldName::from(c.key.as_ref())))
}

/// Gathers row-aligned column arrays into the `fields` struct.
pub(crate) fn fields_struct(
    columns: &[ShreddedColumn],
    column_arrays: Vec<ArrayRef>,
    len: usize,
) -> VortexResult<ArrayRef> {
    Ok(StructArray::try_new(
        field_names(columns),
        column_arrays,
        len,
        Validity::NonNullable,
    )?
    .into_array())
}

/// The columns of a `fields` struct.
pub(crate) fn struct_columns(fields: &ArrayRef) -> Vec<ArrayRef> {
    fields
        .as_opt::<Struct>()
        .map(|s| s.iter_unmasked_fields().cloned().collect())
        .unwrap_or_default()
}

pub(crate) fn make_parts(
    dtype: DType,
    len: usize,
    columns: Arc<[ShreddedColumn]>,
    residual: ArrayRef,
    repeats: Option<ArrayRef>,
    fields: ArrayRef,
) -> ArrayParts<ShreddedMap> {
    let mut slots = ArraySlots::with_capacity(ShreddedMapSlots::COUNT);
    slots.push(Some(residual));
    slots.push(repeats);
    slots.push(Some(fields));
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
        Self::try_new_with_repeats(residual, None, columns, column_arrays)
    }

    /// Like [`try_new`](Self::try_new), with a [`repeats`](ShreddedMapSlots::repeats) hint.
    ///
    /// # Errors
    ///
    /// Returns an error if the parts do not match the layout described on [`ShreddedMap`].
    pub fn try_new_with_repeats(
        residual: ArrayRef,
        repeats: Option<ArrayRef>,
        columns: Vec<ShreddedColumn>,
        column_arrays: Vec<ArrayRef>,
    ) -> VortexResult<ShreddedMapArray> {
        let dtype = residual.dtype().clone();
        let len = residual.len();
        vortex_ensure!(
            columns.len() == column_arrays.len(),
            "column metadata and arrays differ in length"
        );
        let fields = fields_struct(&columns, column_arrays, len)?;
        Array::try_from_parts(make_parts(
            dtype,
            len,
            columns.into(),
            residual,
            repeats,
            fields,
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
        .column_arrays()
        .iter()
        .map(|column| compress_column(column, &mut compress))
        .collect::<VortexResult<Vec<_>>>()?;
    let repeats = array.repeats().map(&mut compress).transpose()?;
    ShreddedMap::try_new_with_repeats(residual, repeats, array.data().columns().to_vec(), columns)
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
            .map(|slot| {
                slot.as_ref()
                    .map(|c| compress_column(c, compress))
                    .transpose()
            })
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

    /// The shredded columns, the fields of [`fields`](ShreddedMapSlots::fields) in key order.
    fn column_arrays(&self) -> Vec<ArrayRef> {
        struct_columns(self.fields())
    }

    /// The column of the `index`-th shredded key.
    fn column_array(&self, index: usize) -> ArrayRef {
        self.fields()
            .as_opt::<Struct>()
            .vortex_expect("ShreddedMap fields must be a struct array")
            .unmasked_field(index)
            .clone()
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
        #[allow(clippy::cast_possible_truncation)]
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
            has_repeats: array.repeats().is_some(),
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
        // Children are the present slots in order, so the fields follow the optional repeats.
        let fields_child = 1 + usize::from(proto.has_repeats);
        vortex_ensure!(
            children.len() == fields_child + 1,
            "ShreddedMapArray expected {} children, found {}",
            fields_child + 1,
            children.len()
        );
        let residual = children.get(0, dtype, len)?;
        let repeats = proto
            .has_repeats
            .then(|| children.get(1, &DType::Bool(Nullability::NonNullable), len))
            .transpose()?;
        let fields = children.get(fields_child, &fields_dtype(map_dtype, &columns)?, len)?;
        Ok(make_parts(
            dtype.clone(),
            len,
            columns,
            residual,
            repeats,
            fields,
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ShreddedMapSlots::NAMES[idx].to_string()
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
