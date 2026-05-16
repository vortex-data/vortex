use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_mask::Mask;

use crate::DomainSpan;

/// A nullable cell in the prototype's row representation.
pub type Cell = Option<i64>;

/// One row in the prototype's row representation. The prototype keeps
/// `Cell` as `Option<i64>` for simplicity; production batches carry
/// arbitrary Vortex `DType` payloads.
pub type Row = Vec<Cell>;

/// A vectorized unit of execution.
///
/// `Batch` carries one Vortex `ArrayRef`, a `DomainSpan` over the
/// producing port's domain, and a per-row `demand: Mask` describing
/// which rows have *real* values vs. *don't-care* placeholders.
///
/// Invariants:
///   `array.len() == span.len() == demand.len()`
///
/// The demand mask is the producer's static commitment about the
/// data in `array`: `demand[i] == true` means row `i` carries real
/// values; `demand[i] == false` means the value at row `i` is
/// undefined (the producer skipped its real work for that row,
/// typically because downstream told it those rows were
/// `NotNeeded`).
///
/// **Row demand never changes the domain.** The span and length
/// stay fixed even when many rows are don't-care — alignment
/// downstream is preserved. Operators that *do* drop rows (Filter,
/// Gather, Repartition) mint a fresh output domain with
/// `Cardinality::Unknown` and a witness back to input rows; that's
/// distinct from row-demand placeholders.
///
/// Most callers don't think about demand: `from_array` defaults to
/// all-true demand, and pass-through operators preserve it. Only
/// sources that skip I/O for `NotNeeded` ranges and consumers that
/// gate kernels on real values need to handle it explicitly.
#[derive(Clone, Debug)]
pub struct Batch {
    span: DomainSpan,
    array: ArrayRef,
    demand: Mask,
    /// Cached `array.nbytes()` computed once at construction.
    /// `nbytes()` is recursive (walks the array tree) and used to
    /// be the dominant cost of `Channel::retained_bytes` /
    /// `has_capacity` — recomputing per call became O(N_batches)
    /// per push. Cache once; it never changes after construction
    /// because `Batch` is immutable.
    estimated_bytes: usize,
}

/// Compute `array.nbytes()` once and clamp into `usize`.
fn compute_estimated_bytes(array: &ArrayRef) -> usize {
    usize::try_from(array.nbytes()).unwrap_or(usize::MAX)
}

impl Batch {
    /// Build a `Batch` from a Vortex `ArrayRef`, defaulting demand
    /// to all-true (every row carries real values).
    pub fn from_array(span: DomainSpan, array: ArrayRef) -> Self {
        let len = array.len();
        debug_assert_eq!(
            u64::try_from(len).unwrap_or(u64::MAX),
            span.len(),
            "batch row count must match span length"
        );
        let estimated_bytes = compute_estimated_bytes(&array);
        Self {
            span,
            array,
            demand: Mask::new_true(len),
            estimated_bytes,
        }
    }

    /// Build a `Batch` with an explicit demand mask. Use this when
    /// some rows of `array` are placeholders / don't-care and the
    /// producer wants downstream consumers to skip them.
    pub fn with_demand(span: DomainSpan, array: ArrayRef, demand: Mask) -> Self {
        let len = array.len();
        debug_assert_eq!(
            u64::try_from(len).unwrap_or(u64::MAX),
            span.len(),
            "batch row count must match span length"
        );
        debug_assert_eq!(
            demand.len(),
            len,
            "batch demand mask must match array length"
        );
        let estimated_bytes = compute_estimated_bytes(&array);
        Self {
            span,
            array,
            demand,
            estimated_bytes,
        }
    }

    /// Build a placeholder batch: all-false demand, array is a
    /// cheap constant-zero of `dtype`. Used by sources to emit
    /// don't-care row ranges without paying any real I/O or decode
    /// cost while preserving the output domain.
    pub fn placeholder(span: DomainSpan, dtype: DType) -> Self {
        let len = usize::try_from(span.len()).unwrap_or(usize::MAX);
        let scalar = Scalar::null(dtype.as_nullable());
        let array = ConstantArray::new(scalar, len).into_array();
        let estimated_bytes = compute_estimated_bytes(&array);
        Self {
            span,
            array,
            demand: Mask::new_false(len),
            estimated_bytes,
        }
    }

    /// Build a single-column `i64` `Batch` from an iterator of values.
    pub fn from_values(start: u64, values: impl IntoIterator<Item = i64>) -> Self {
        let values: Vec<i64> = values.into_iter().collect();
        let len_u = values.len();
        let len = u64::try_from(len_u).unwrap_or(u64::MAX);
        let column: ArrayRef = PrimitiveArray::from_iter(values).into_array();
        let array = build_struct(vec![("col0", column)]);
        let estimated_bytes = compute_estimated_bytes(&array);
        Self {
            span: DomainSpan::new(start, len),
            array,
            demand: Mask::new_true(len_u),
            estimated_bytes,
        }
    }

    /// Build a `Batch` from row-major `Vec<Row>`. Each cell is
    /// `Option<i64>`; a `None` produces a null at that position.
    pub fn from_rows(start: u64, rows: Vec<Row>) -> Self {
        let row_count = rows.len();
        let column_count = rows.first().map_or(0, Vec::len);
        let mut columns: Vec<Vec<Option<i64>>> = vec![Vec::with_capacity(row_count); column_count];
        for row in rows {
            for (column_index, column) in columns.iter_mut().enumerate() {
                column.push(row.get(column_index).copied().flatten());
            }
        }
        let fields: Vec<(String, ArrayRef)> = columns
            .into_iter()
            .enumerate()
            .map(|(idx, values)| {
                let array = if values.iter().all(Option::is_some) {
                    let typed: Vec<i64> = values.into_iter().map(Option::unwrap_or_default).collect();
                    PrimitiveArray::from_iter(typed).into_array()
                } else {
                    PrimitiveArray::from_option_iter::<i64, _>(values).into_array()
                };
                (format!("col{idx}"), array)
            })
            .collect();
        let fields_ref: Vec<(&str, ArrayRef)> =
            fields.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
        let array = build_struct(fields_ref);
        let estimated_bytes = compute_estimated_bytes(&array);
        Self {
            span: DomainSpan::new(start, u64::try_from(row_count).unwrap_or(u64::MAX)),
            array,
            demand: Mask::new_true(row_count),
            estimated_bytes,
        }
    }

    /// Convenience alias kept for prototype backwards compatibility.
    /// Production batches do not distinguish "lazy" rows; Vortex
    /// arrays are inherently lazy.
    pub fn from_lazy_rows(start: u64, rows: Vec<Row>) -> Self {
        Self::from_rows(start, rows)
    }

    /// Build a `Batch` from column-major data. Each inner `Vec<Cell>`
    /// is one column.
    pub fn from_lazy_columns(start: u64, columns: Vec<Vec<Cell>>) -> Self {
        let row_count = columns.first().map_or(0, Vec::len);
        let len = u64::try_from(row_count).unwrap_or(u64::MAX);
        let fields_owned: Vec<(String, ArrayRef)> = columns
            .into_iter()
            .enumerate()
            .map(|(idx, values)| {
                let array = if values.iter().all(Option::is_some) {
                    let typed: Vec<i64> = values.into_iter().map(Option::unwrap_or_default).collect();
                    PrimitiveArray::from_iter(typed).into_array()
                } else {
                    PrimitiveArray::from_option_iter::<i64, _>(values).into_array()
                };
                (format!("col{idx}"), array)
            })
            .collect();
        let fields_ref: Vec<(&str, ArrayRef)> = fields_owned
            .iter()
            .map(|(n, a)| (n.as_str(), a.clone()))
            .collect();
        let array = build_struct(fields_ref);
        let estimated_bytes = compute_estimated_bytes(&array);
        Self {
            span: DomainSpan::new(start, len),
            array,
            demand: Mask::new_true(row_count),
            estimated_bytes,
        }
    }

    pub const fn span(&self) -> DomainSpan {
        self.span
    }

    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    pub fn into_array(self) -> ArrayRef {
        self.array
    }

    /// Demand mask: which rows in `array` carry real values
    /// (`true`) vs. don't-care placeholders (`false`).
    pub fn demand(&self) -> &Mask {
        &self.demand
    }

    /// True if every row in the batch has real values.
    pub fn demand_all_true(&self) -> bool {
        self.demand.all_true()
    }

    /// True if every row in the batch is a don't-care placeholder.
    pub fn demand_all_false(&self) -> bool {
        self.demand.all_false()
    }

    pub fn dtype(&self) -> &DType {
        self.array.dtype()
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.len() == 0
    }

    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    /// Extract a column as `Vec<Option<i64>>`. Used by prototype
    /// operators that consume row-by-row.
    pub fn column_values(&self, column: usize) -> Vec<Cell> {
        let row_count = self.array.len();
        let Some(struct_array) = downcast_struct(&self.array) else {
            return vec![None; row_count];
        };
        let fields = struct_array.unmasked_fields();
        let Some(field) = fields.get(column) else {
            return vec![None; row_count];
        };
        primitive_to_cells(field, row_count)
    }

    /// Extract the first column as a `Vec<i64>`, dropping nulls and
    /// using `0` placeholders for null cells. Used by the prototype's
    /// row-extraction helpers.
    pub fn first_column_values(&self) -> Vec<i64> {
        self.column_values(0).into_iter().flatten().collect()
    }

    /// Project a subset of columns into a new `Batch`. Columns are
    /// selected by integer position in the underlying struct.
    pub fn project_columns(&self, columns: &[usize]) -> Self {
        let row_count = self.array.len();
        let Some(struct_array) = downcast_struct(&self.array) else {
            // Should not happen for prototype batches but stay safe.
            return self.clone();
        };
        let fields = struct_array.unmasked_fields();
        let projected: Vec<(String, ArrayRef)> = columns
            .iter()
            .enumerate()
            .map(|(out_idx, in_idx)| {
                let column = fields
                    .get(*in_idx)
                    .cloned()
                    .unwrap_or_else(|| empty_i64_column(row_count));
                (format!("col{out_idx}"), column)
            })
            .collect();
        let projected_ref: Vec<(&str, ArrayRef)> = projected
            .iter()
            .map(|(n, a)| (n.as_str(), a.clone()))
            .collect();
        let array = build_struct(projected_ref);
        let estimated_bytes = compute_estimated_bytes(&array);
        Self {
            span: self.span,
            array,
            demand: self.demand.clone(),
            estimated_bytes,
        }
    }

    /// Materialize the batch back to row-major form.
    pub fn to_rows(&self) -> Vec<Row> {
        let row_count = self.array.len();
        let Some(struct_array) = downcast_struct(&self.array) else {
            return Vec::new();
        };
        let fields = struct_array.unmasked_fields();
        let column_data: Vec<Vec<Cell>> = fields
            .iter()
            .map(|f| primitive_to_cells(f, row_count))
            .collect();
        (0..row_count)
            .map(|row_index| {
                column_data
                    .iter()
                    .map(|column| column.get(row_index).copied().unwrap_or_default())
                    .collect()
            })
            .collect()
    }

    pub fn into_rows(self) -> Vec<Row> {
        self.to_rows()
    }
}

impl PartialEq for Batch {
    fn eq(&self, other: &Self) -> bool {
        // Equality compares row contents materialized to rows. Used
        // mostly by tests; not a hot path.
        self.span == other.span && self.to_rows() == other.to_rows()
    }
}

impl Eq for Batch {}

fn build_struct(fields: Vec<(&str, ArrayRef)>) -> ArrayRef {
    if fields.is_empty() {
        return StructArray::new_fieldless_with_len(0).into_array();
    }
    let len = fields[0].1.len();
    let names: FieldNames =
        FieldNames::from(fields.iter().map(|(n, _)| FieldName::from(*n)).collect::<Vec<_>>());
    let dtypes: Vec<DType> = fields.iter().map(|(_, a)| a.dtype().clone()).collect();
    let struct_fields = StructFields::new(names, dtypes);
    let arrays: Vec<ArrayRef> = fields.into_iter().map(|(_, a)| a).collect();
    StructArray::try_new_with_dtype(arrays, struct_fields, len, Validity::NonNullable)
        .map(IntoArray::into_array)
        .unwrap_or_else(|_| StructArray::new_fieldless_with_len(len).into_array())
}

fn downcast_struct(array: &ArrayRef) -> Option<Array<Struct>> {
    array.clone().try_downcast::<Struct>().ok()
}

fn primitive_to_cells(field: &ArrayRef, row_count: usize) -> Vec<Cell> {
    // Try to downcast to a primitive i64 array. If the dtype isn't
    // primitive i64 we fall back to all-null; the prototype only
    // produces i64 columns.
    let Ok(prim) = field.clone().try_downcast::<Primitive>() else {
        return vec![None; row_count];
    };
    if prim.ptype() != PType::I64 {
        return vec![None; row_count];
    }
    let buffer = prim.to_buffer::<i64>();
    let Ok(validity) = prim.validity() else {
        return vec![None; row_count];
    };
    (0..row_count)
        .map(|i| {
            let value = buffer.get(i).copied().unwrap_or_default();
            match validity.is_valid(i) {
                Ok(true) => Some(value),
                _ => None,
            }
        })
        .collect()
}

fn empty_i64_column(len: usize) -> ArrayRef {
    let validity = Validity::AllInvalid;
    let buffer = vec![0i64; len];
    PrimitiveArray::new(buffer, validity).into_array()
}
