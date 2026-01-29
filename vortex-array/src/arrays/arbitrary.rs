// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

// ============================================================================
// Constraint Types
// ============================================================================

/// Kinds of constraints that can be applied to arbitrary array generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
    /// Values must be strictly increasing (each > previous)
    StrictlySorted,
    /// Values must be monotonically increasing (each >= previous)
    Sorted,
    /// First value must be zero
    StartsAtZero,
    /// Values must be < some upper bound
    BoundedAbove,
    /// Values must be >= some lower bound
    BoundedBelow,
    /// Values must fit in a specific bit width
    BitWidthBounded,
    /// Array must be non-nullable
    NonNullable,
    /// Must be unsigned integer type
    Unsigned,
    /// Must be integer type (not float)
    IntegerOnly,
}

/// Ordering constraints for array values.
#[derive(Clone, Debug, Default)]
pub struct OrderingConstraint {
    /// Values must be strictly increasing (each > previous)
    pub strictly_sorted: bool,
    /// Values must be monotonically increasing (each >= previous)
    pub sorted: bool,
    /// First value must equal this
    pub starts_at: Option<u64>,
}

/// Value range constraints.
#[derive(Clone, Debug, Default)]
pub struct BoundConstraint {
    /// All values must be < upper_bound
    pub upper_bound: Option<u64>,
    /// All values must be >= lower_bound
    pub lower_bound: Option<u64>,
    /// Soft target for maximum value (used for sorted arrays)
    pub target_max: Option<u64>,
    /// Values must fit in this many bits
    pub bit_width: Option<u8>,
}

/// Type constraints.
#[derive(Clone, Debug, Default)]
pub struct TypeConstraint {
    /// Must be unsigned integer
    pub unsigned: bool,
    /// Must be integer (not float)
    pub integer_only: bool,
    /// If Some, must be one of these ptypes
    pub allowed_ptypes: Option<Vec<PType>>,
}

/// Combined constraints for arbitrary array generation.
#[derive(Clone, Debug, Default)]
pub struct ArrayConstraints {
    /// Ordering constraints (sorted, strictly sorted, etc.)
    pub ordering: OrderingConstraint,
    /// Value range constraints (bounds, bit width, etc.)
    pub bounds: BoundConstraint,
    /// Type constraints (unsigned, integer only, etc.)
    pub type_constraint: TypeConstraint,
    /// Must be non-nullable
    pub non_nullable: bool,
    /// Must be valid UTF-8 (for binary data)
    pub valid_utf8: bool,
}

impl ArrayConstraints {
    /// Check if these constraints require sorted values.
    pub fn requires_sorted(&self) -> bool {
        self.ordering.sorted || self.ordering.strictly_sorted
    }

    /// Check if these constraints require strictly sorted values.
    pub fn requires_strictly_sorted(&self) -> bool {
        self.ordering.strictly_sorted
    }

    /// Check if these constraints are satisfied by an encoding's capabilities.
    pub fn is_satisfied_by(&self, capabilities: &[ConstraintKind]) -> bool {
        // Check ordering constraints
        if self.ordering.strictly_sorted && !capabilities.contains(&ConstraintKind::StrictlySorted)
        {
            return false;
        }
        if self.ordering.sorted
            && !capabilities.contains(&ConstraintKind::Sorted)
            && !capabilities.contains(&ConstraintKind::StrictlySorted)
        {
            return false;
        }
        if self.ordering.starts_at.is_some()
            && !capabilities.contains(&ConstraintKind::StartsAtZero)
        {
            return false;
        }

        // Check bound constraints
        if self.bounds.upper_bound.is_some()
            && !capabilities.contains(&ConstraintKind::BoundedAbove)
        {
            return false;
        }
        if self.bounds.lower_bound.is_some()
            && !capabilities.contains(&ConstraintKind::BoundedBelow)
        {
            return false;
        }
        if self.bounds.bit_width.is_some()
            && !capabilities.contains(&ConstraintKind::BitWidthBounded)
        {
            return false;
        }

        // Check type constraints
        if self.type_constraint.unsigned && !capabilities.contains(&ConstraintKind::Unsigned) {
            return false;
        }
        if self.type_constraint.integer_only && !capabilities.contains(&ConstraintKind::IntegerOnly)
        {
            return false;
        }

        // Check nullability
        if self.non_nullable && !capabilities.contains(&ConstraintKind::NonNullable) {
            return false;
        }

        true
    }
}

// ============================================================================
// Constrained Generation Trait
// ============================================================================

/// Generator function type for constrained array generation.
pub type ConstrainedGenerator =
    fn(&mut Unstructured, Option<usize>, &DType, &ArrayConstraints) -> Result<ArrayRef>;

/// Trait for encodings that support constrained arbitrary generation.
pub trait ArbitraryConstrained {
    /// Constraints this encoding can satisfy.
    ///
    /// For leaf encodings: constraints it can generate directly.
    /// For wrapping encodings: constraints it can satisfy by wrapping a constrained child.
    fn can_generate() -> &'static [ConstraintKind];

    /// Generate an arbitrary array satisfying the given constraints.
    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef>;
}

// ============================================================================
// Constrained Primitive Generation
// ============================================================================

/// Primitive array can generate all constraint kinds (it's the base case).
pub const PRIMITIVE_CAN_GENERATE: &[ConstraintKind] = &[
    ConstraintKind::StrictlySorted,
    ConstraintKind::Sorted,
    ConstraintKind::StartsAtZero,
    ConstraintKind::BoundedAbove,
    ConstraintKind::BoundedBelow,
    ConstraintKind::BitWidthBounded,
    ConstraintKind::NonNullable,
    ConstraintKind::Unsigned,
    ConstraintKind::IntegerOnly,
];

/// Generate a constrained primitive array.
pub fn random_primitive_constrained(
    u: &mut Unstructured,
    len: Option<usize>,
    dtype: &DType,
    constraints: &ArrayConstraints,
) -> Result<ArrayRef> {
    let DType::Primitive(ptype, nullability) = dtype else {
        // Fall back to unconstrained for non-primitive types
        return random_array(u, dtype, len);
    };

    let len = len.unwrap_or(u.int_in_range(0..=100)?);
    let nullability = if constraints.non_nullable {
        Nullability::NonNullable
    } else {
        *nullability
    };

    if constraints.requires_sorted() {
        random_sorted_primitive(u, len, *ptype, nullability, constraints)
    } else if constraints.bounds.upper_bound.is_some() || constraints.bounds.lower_bound.is_some() {
        random_bounded_primitive(u, len, *ptype, nullability, constraints)
    } else {
        // Unconstrained
        let dtype = DType::Primitive(*ptype, nullability);
        random_array(u, &dtype, Some(len))
    }
}

/// Generate a sorted primitive array using the increments approach.
fn random_sorted_primitive(
    u: &mut Unstructured,
    len: usize,
    ptype: PType,
    nullability: Nullability,
    constraints: &ArrayConstraints,
) -> Result<ArrayRef> {
    if len == 0 {
        return Ok(PrimitiveArray::new(
            Buffer::<u64>::empty(),
            if nullability == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
        .reinterpret_cast(ptype)
        .into_array());
    }

    // Determine bounds for increments
    let min_increment: u64 = if constraints.ordering.strictly_sorted {
        1
    } else {
        0
    };

    // Calculate max increment based on target_max and upper_bound
    let effective_max = constraints
        .bounds
        .target_max
        .or(constraints.bounds.upper_bound)
        .unwrap_or(1000);

    let start_value = constraints.ordering.starts_at.unwrap_or(0);

    // Calculate max possible increment to stay within bounds
    // If we have len values starting at start_value, and each increment is at least min_increment,
    // the final value would be: start_value + sum(increments)
    // We want: start_value + sum(increments) <= effective_max
    // With len increments averaging max_increment: start_value + len * max_increment <= effective_max
    let available_range = effective_max.saturating_sub(start_value);
    let max_increment = if len > 0 {
        (available_range / len as u64).max(min_increment)
    } else {
        min_increment
    };

    // Generate increments and compute cumulative sum
    let mut values: Vec<u64> = Vec::with_capacity(len);
    let mut current = start_value;

    for _ in 0..len {
        values.push(current);
        let increment = u.int_in_range(min_increment..=max_increment)?;
        current = current.saturating_add(increment);
    }

    // Convert to target ptype
    let validity = random_validity(u, nullability, len)?;
    let array = convert_u64_vec_to_ptype(values, ptype, validity);
    Ok(array)
}

/// Generate a bounded primitive array.
fn random_bounded_primitive(
    u: &mut Unstructured,
    len: usize,
    ptype: PType,
    nullability: Nullability,
    constraints: &ArrayConstraints,
) -> Result<ArrayRef> {
    let lower = constraints.bounds.lower_bound.unwrap_or(0);
    let upper = constraints
        .bounds
        .upper_bound
        .unwrap_or_else(|| ptype_max_value(ptype));

    // Generate values within bounds
    let values: Vec<u64> = (0..len)
        .map(|_| u.int_in_range(lower..=upper.saturating_sub(1).max(lower)))
        .collect::<Result<Vec<_>>>()?;

    let validity = random_validity(u, nullability, len)?;
    let array = convert_u64_vec_to_ptype(values, ptype, validity);
    Ok(array)
}

/// Convert a Vec<u64> to a PrimitiveArray of the given ptype.
#[allow(clippy::cast_possible_truncation)]
fn convert_u64_vec_to_ptype(values: Vec<u64>, ptype: PType, validity: Validity) -> ArrayRef {
    match ptype {
        PType::U8 => {
            let v: Vec<u8> = values.into_iter().map(|x| x as u8).collect();
            PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
        }
        PType::U16 => {
            let v: Vec<u16> = values.into_iter().map(|x| x as u16).collect();
            PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
        }
        PType::U32 => {
            let v: Vec<u32> = values.into_iter().map(|x| x as u32).collect();
            PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
        }
        PType::U64 => PrimitiveArray::new(Buffer::copy_from(values), validity).into_array(),
        PType::I8 => {
            let v: Vec<i8> = values.into_iter().map(|x| x as i8).collect();
            PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
        }
        PType::I16 => {
            let v: Vec<i16> = values.into_iter().map(|x| x as i16).collect();
            PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
        }
        PType::I32 => {
            let v: Vec<i32> = values.into_iter().map(|x| x as i32).collect();
            PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
        }
        PType::I64 => {
            let v: Vec<i64> = values.into_iter().map(|x| x as i64).collect();
            PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
        }
        PType::F16 | PType::F32 | PType::F64 => {
            // For floats, just cast - sorted property preserved
            let v: Vec<f64> = values.into_iter().map(|x| x as f64).collect();
            match ptype {
                PType::F32 => {
                    let v: Vec<f32> = v.into_iter().map(|x| x as f32).collect();
                    PrimitiveArray::new(Buffer::copy_from(v), validity).into_array()
                }
                PType::F64 => PrimitiveArray::new(Buffer::copy_from(v), validity).into_array(),
                PType::F16 => {
                    let v: Vec<u16> = v
                        .into_iter()
                        .map(|x| vortex_dtype::half::f16::from_f64(x).to_bits())
                        .collect();
                    PrimitiveArray::new(Buffer::copy_from(v), validity)
                        .reinterpret_cast(PType::F16)
                        .into_array()
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Get the maximum value for a ptype (for bounding).
fn ptype_max_value(ptype: PType) -> u64 {
    match ptype {
        PType::U8 => u8::MAX as u64,
        PType::U16 => u16::MAX as u64,
        PType::U32 => u32::MAX as u64,
        PType::U64 => u64::MAX,
        PType::I8 => i8::MAX as u64,
        PType::I16 => i16::MAX as u64,
        PType::I32 => i32::MAX as u64,
        PType::I64 => i64::MAX as u64,
        PType::F16 => u16::MAX as u64,
        PType::F32 => u32::MAX as u64,
        PType::F64 => u64::MAX,
    }
}

// ============================================================================
// Dispatcher
// ============================================================================

/// Generate a random array satisfying the given constraints.
/// May choose any compatible encoding randomly.
pub fn arbitrary_constrained_array(
    u: &mut Unstructured,
    len: Option<usize>,
    dtype: &DType,
    constraints: &ArrayConstraints,
) -> Result<ArrayRef> {
    // Use the recursive version with default max depth
    arbitrary_constrained_array_with_depth(u, len, dtype, constraints, 8)
}

/// Generate a random array with controlled nesting depth.
/// This enables deeply nested array structures for fuzz testing.
pub fn arbitrary_constrained_array_with_depth(
    u: &mut Unstructured,
    len: Option<usize>,
    dtype: &DType,
    constraints: &ArrayConstraints,
    max_depth: usize,
) -> Result<ArrayRef> {
    let len = len.unwrap_or(u.int_in_range(1..=20)?);

    // Base case: no more depth allowed, use primitive
    if max_depth == 0 {
        return random_primitive_constrained(u, Some(len), dtype, constraints);
    }

    // Only wrap integer types in complex encodings
    let DType::Primitive(ptype, nullability) = dtype else {
        return random_primitive_constrained(u, Some(len), dtype, constraints);
    };

    // Choose encoding based on constraints and randomness
    // Higher depth = more likely to wrap in DictArray
    // 0 = Primitive (base case)
    // 1 = SequenceArray (if sorted constraints)
    // 2 = DictArray wrapper (recursive nesting)
    let encoding_choice = if constraints.requires_strictly_sorted() || constraints.requires_sorted()
    {
        // For sorted constraints, prefer SequenceArray or Primitive
        u.int_in_range(0..=1)?
    } else {
        // Bias toward DictArray when we have depth budget
        // 50% chance of DictArray when depth > 2
        if max_depth > 2 && u.arbitrary::<bool>()? {
            2 // DictArray
        } else {
            u.int_in_range(0..=2)?
        }
    };

    match encoding_choice {
        0 => {
            // Base case: Primitive
            random_primitive_constrained(u, Some(len), dtype, constraints)
        }
        1 if constraints.requires_strictly_sorted() => {
            // SequenceArray for strictly sorted (zero storage!)
            create_sequence_array(u, len, *ptype, *nullability, constraints)
        }
        1 => {
            // Primitive for non-sorted
            random_primitive_constrained(u, Some(len), dtype, constraints)
        }
        2 => {
            // DictArray: codes index into values
            // Generate small values array, then bounded codes
            let values_len = u.int_in_range(1..=len.clamp(1, 10))?;

            // Recursively generate values with reduced depth
            let values = arbitrary_constrained_array_with_depth(
                u,
                Some(values_len),
                dtype,
                &ArrayConstraints::default(),
                max_depth - 1,
            )?;

            // Generate bounded codes
            let codes_ptype = PType::min_unsigned_ptype_for_value((values_len - 1) as u64);
            let codes_dtype = DType::Primitive(
                codes_ptype,
                if constraints.non_nullable {
                    Nullability::NonNullable
                } else {
                    *nullability
                },
            );
            let codes_constraints = ArrayConstraints {
                bounds: BoundConstraint {
                    lower_bound: Some(0),
                    upper_bound: Some(values_len as u64),
                    ..Default::default()
                },
                non_nullable: constraints.non_nullable,
                ..Default::default()
            };

            // Recursively generate codes with reduced depth
            let codes = arbitrary_constrained_array_with_depth(
                u,
                Some(len),
                &codes_dtype,
                &codes_constraints,
                max_depth - 1,
            )?;

            Ok(super::DictArray::try_new(codes, values)
                .vortex_expect("DictArray creation in arbitrary")
                .into_array())
        }
        _ => random_primitive_constrained(u, Some(len), dtype, constraints),
    }
}

/// Create a SequenceArray for sorted/strictly-sorted constraints.
/// Note: SequenceArray is in vortex-sequence crate, so we fall back to sorted primitive.
fn create_sequence_array(
    u: &mut Unstructured,
    len: usize,
    ptype: PType,
    nullability: Nullability,
    constraints: &ArrayConstraints,
) -> Result<ArrayRef> {
    // SequenceArray is in vortex-sequence crate, not vortex-array.
    // Fall back to sorted primitive generation which satisfies the same constraints.
    random_primitive_constrained(
        u,
        Some(len),
        &DType::Primitive(ptype, nullability),
        constraints,
    )
}

use super::BoolArray;
use super::ChunkedArray;
use super::NullArray;
use super::PrimitiveArray;
use super::StructArray;
use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::builders::ArrayBuilder;
use crate::builders::DecimalBuilder;
use crate::builders::FixedSizeListBuilder;
use crate::builders::ListViewBuilder;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::scalar::Scalar;
use crate::scalar::arbitrary::random_scalar;
use crate::validity::Validity;

/// A wrapper type to implement `Arbitrary` for `ArrayRef`.
#[derive(Clone, Debug)]
pub struct ArbitraryArray(pub ArrayRef);

impl<'a> Arbitrary<'a> for ArbitraryArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype = u.arbitrary()?;
        Self::arbitrary_with(u, None, &dtype)
    }
}

impl ArbitraryArray {
    pub fn arbitrary_with(u: &mut Unstructured, len: Option<usize>, dtype: &DType) -> Result<Self> {
        random_array(u, dtype, len).map(ArbitraryArray)
    }
}

fn split_number_into_parts(n: usize, parts: usize) -> Vec<usize> {
    let reminder = n % parts;
    let division = (n - reminder) / parts;
    iter::repeat_n(division, parts - reminder)
        .chain(iter::repeat_n(division + 1, reminder))
        .collect()
}

/// Creates a random array with a random number of chunks.
fn random_array(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<ArrayRef> {
    let num_chunks = u.int_in_range(1..=3)?;
    let chunk_lens = len.map(|l| split_number_into_parts(l, num_chunks));
    let mut chunks = (0..num_chunks)
        .map(|i| {
            let chunk_len = chunk_lens.as_ref().map(|c| c[i]);
            random_array_chunk(u, dtype, chunk_len)
        })
        .collect::<Result<Vec<_>>>()?;

    if chunks.len() == 1 {
        Ok(chunks.remove(0))
    } else {
        let dtype = chunks[0].dtype().clone();
        Ok(ChunkedArray::try_new(chunks, dtype)
            .vortex_expect("operation should succeed in arbitrary impl")
            .into_array())
    }
}

/// Creates a random array chunk.
fn random_array_chunk(
    u: &mut Unstructured<'_>,
    dtype: &DType,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    match dtype {
        DType::Null => Ok(NullArray::new(
            chunk_len
                .map(Ok)
                .unwrap_or_else(|| u.int_in_range(0..=100))?,
        )
        .into_array()),
        DType::Bool(n) => random_bool(u, *n, chunk_len),
        DType::Primitive(ptype, n) => match ptype {
            PType::U8 => random_primitive::<u8>(u, *n, chunk_len),
            PType::U16 => random_primitive::<u16>(u, *n, chunk_len),
            PType::U32 => random_primitive::<u32>(u, *n, chunk_len),
            PType::U64 => random_primitive::<u64>(u, *n, chunk_len),
            PType::I8 => random_primitive::<i8>(u, *n, chunk_len),
            PType::I16 => random_primitive::<i16>(u, *n, chunk_len),
            PType::I32 => random_primitive::<i32>(u, *n, chunk_len),
            PType::I64 => random_primitive::<i64>(u, *n, chunk_len),
            PType::F16 => Ok(random_primitive::<u16>(u, *n, chunk_len)?
                .to_primitive()
                .reinterpret_cast(PType::F16)
                .into_array()),
            PType::F32 => random_primitive::<f32>(u, *n, chunk_len),
            PType::F64 => random_primitive::<f64>(u, *n, chunk_len),
        },
        d @ DType::Decimal(decimal, n) => {
            let elem_len = chunk_len.unwrap_or(u.int_in_range(0..=20)?);
            match_each_decimal_value_type!(DecimalType::smallest_decimal_value_type(decimal), |D| {
                let mut builder = DecimalBuilder::new::<D>(*decimal, *n);
                for _i in 0..elem_len {
                    let random_decimal = random_scalar(u, d)?;
                    builder.append_scalar(&random_decimal).vortex_expect(
                        "was somehow unable to append a decimal to a decimal builder",
                    );
                }
                Ok(builder.finish())
            })
        }
        DType::Utf8(n) => random_string(u, *n, chunk_len),
        DType::Binary(n) => random_bytes(u, *n, chunk_len),
        DType::Struct(sdt, n) => {
            let first_array = sdt
                .fields()
                .next()
                .map(|d| random_array(u, &d, chunk_len))
                .transpose()?;
            let resolved_len = first_array
                .as_ref()
                .map(|a| a.len())
                .or(chunk_len)
                .map(Ok)
                .unwrap_or_else(|| u.int_in_range(0..=100))?;
            let children = first_array
                .into_iter()
                .map(Ok)
                .chain(
                    sdt.fields()
                        .skip(1)
                        .map(|d| random_array(u, &d, Some(resolved_len))),
                )
                .collect::<Result<Vec<_>>>()?;
            Ok(StructArray::try_new(
                sdt.names().clone(),
                children,
                resolved_len,
                random_validity(u, *n, resolved_len)?,
            )
            .vortex_expect("operation should succeed in arbitrary impl")
            .into_array())
        }
        DType::List(elem_dtype, null) => random_list(u, elem_dtype, *null, chunk_len),
        DType::FixedSizeList(elem_dtype, list_size, null) => {
            random_fixed_size_list(u, elem_dtype, *list_size, *null, chunk_len)
        }
        DType::Extension(..) => {
            todo!("Extension arrays are not implemented")
        }
    }
}

/// Creates a random fixed-size list array.
///
/// If the `chunk_len` is specified, the length of the array will be equal to the chunk length.
fn random_fixed_size_list(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    list_size: u32,
    null: Nullability,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    let array_length = chunk_len.unwrap_or(u.int_in_range(0..=20)?);

    let mut builder =
        FixedSizeListBuilder::with_capacity(elem_dtype.clone(), list_size, null, array_length);

    for _ in 0..array_length {
        if null == Nullability::Nullable && u.arbitrary::<bool>()? {
            builder.append_null();
        } else {
            builder
                .append_value(random_list_scalar(u, elem_dtype, list_size, null)?.as_list())
                .vortex_expect("can append value");
        }
    }

    Ok(builder.finish())
}

/// Creates a random list array.
///
/// If the `chunk_len` is specified, the length of the array will be equal to the chunk length.
fn random_list(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    null: Nullability,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    match u.int_in_range(0..=5)? {
        0 => random_list_with_offset_type::<i16>(u, elem_dtype, null, chunk_len),
        1 => random_list_with_offset_type::<i32>(u, elem_dtype, null, chunk_len),
        2 => random_list_with_offset_type::<i64>(u, elem_dtype, null, chunk_len),
        3 => random_list_with_offset_type::<u16>(u, elem_dtype, null, chunk_len),
        4 => random_list_with_offset_type::<u32>(u, elem_dtype, null, chunk_len),
        5 => random_list_with_offset_type::<u64>(u, elem_dtype, null, chunk_len),
        _ => unreachable!("int_in_range returns a value in the above range"),
    }
}

/// Creates a random list array with the given [`IntegerPType`] for the internal offsets child.
///
/// If the `chunk_len` is specified, the length of the array will be equal to the chunk length.
fn random_list_with_offset_type<O: IntegerPType>(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    null: Nullability,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    let array_length = chunk_len.unwrap_or(u.int_in_range(0..=20)?);

    let mut builder = ListViewBuilder::<O, O>::with_capacity(elem_dtype.clone(), null, 20, 10);

    for _ in 0..array_length {
        if null == Nullability::Nullable && u.arbitrary::<bool>()? {
            builder.append_null();
        } else {
            let list_size = u.int_in_range(0..=20)?;
            builder
                .append_value(random_list_scalar(u, elem_dtype, list_size, null)?.as_list())
                .vortex_expect("can append value");
        }
    }

    Ok(builder.finish())
}

/// Creates a random list scalar with the specified list size.
fn random_list_scalar(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    list_size: u32,
    null: Nullability,
) -> Result<Scalar> {
    let elems = (0..list_size)
        .map(|_| random_scalar(u, elem_dtype))
        .collect::<Result<Vec<_>>>()?;
    Ok(Scalar::list(elem_dtype.clone(), elems, null))
}

fn random_string(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    match nullability {
        Nullability::NonNullable => {
            let v = arbitrary_vec_of_len::<String>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_vec(v, DType::Utf8(Nullability::NonNullable)).into_array(),
                1 => VarBinViewArray::from_iter_str(v).into_array(),
                _ => unreachable!(),
            })
        }
        Nullability::Nullable => {
            let v = arbitrary_vec_of_len::<Option<String>>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_iter(v, DType::Utf8(Nullability::Nullable)).into_array(),
                1 => VarBinViewArray::from_iter_nullable_str(v).into_array(),
                _ => unreachable!(),
            })
        }
    }
}

fn random_bytes(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    match nullability {
        Nullability::NonNullable => {
            let v = arbitrary_vec_of_len::<Vec<u8>>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_vec(v, DType::Binary(Nullability::NonNullable)).into_array(),
                1 => VarBinViewArray::from_iter_bin(v).into_array(),
                _ => unreachable!(),
            })
        }
        Nullability::Nullable => {
            let v = arbitrary_vec_of_len::<Option<Vec<u8>>>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_iter(v, DType::Binary(Nullability::Nullable)).into_array(),
                1 => VarBinViewArray::from_iter_nullable_bin(v).into_array(),
                _ => unreachable!(),
            })
        }
    }
}

fn random_primitive<'a, T: Arbitrary<'a> + NativePType>(
    u: &mut Unstructured<'a>,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    let v = arbitrary_vec_of_len::<T>(u, len)?;
    let validity = random_validity(u, nullability, v.len())?;
    Ok(PrimitiveArray::new(Buffer::copy_from(v), validity).into_array())
}

fn random_bool(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    let v = arbitrary_vec_of_len(u, len)?;
    let validity = random_validity(u, nullability, v.len())?;
    Ok(BoolArray::new(BitBuffer::from(v), validity).into_array())
}

pub fn random_validity(
    u: &mut Unstructured,
    nullability: Nullability,
    len: usize,
) -> Result<Validity> {
    match nullability {
        Nullability::NonNullable => Ok(Validity::NonNullable),
        Nullability::Nullable => Ok(match u.int_in_range(0..=2)? {
            0 => Validity::AllValid,
            1 => Validity::AllInvalid,
            2 => Validity::from_iter(arbitrary_vec_of_len::<bool>(u, Some(len))?),
            _ => unreachable!(),
        }),
    }
}

fn arbitrary_vec_of_len<'a, T: Arbitrary<'a>>(
    u: &mut Unstructured<'a>,
    len: Option<usize>,
) -> Result<Vec<T>> {
    len.map(|l| (0..l).map(|_| T::arbitrary(u)).collect::<Result<Vec<_>>>())
        .unwrap_or_else(|| Vec::<T>::arbitrary(u))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;
    use crate::display::DisplayOptions;

    /// Count nesting depth of an array
    fn count_depth(array: &dyn Array, current: usize) -> usize {
        let child_depths: Vec<usize> = array
            .children()
            .iter()
            .map(|c| count_depth(c.as_ref(), current + 1))
            .collect();
        child_depths.into_iter().max().unwrap_or(current)
    }

    #[test]
    fn test_arbitrary_constrained_produces_nested_arrays() {
        // Use different seeds to get variety
        for seed_offset in 0..10 {
            let seed: Vec<u8> = (seed_offset * 100..(seed_offset * 100 + 5000))
                .map(|i| (i % 256) as u8)
                .collect();
            let mut u = Unstructured::new(&seed);

            let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
            let constraints = ArrayConstraints::default();

            let array =
                arbitrary_constrained_array_with_depth(&mut u, Some(10), &dtype, &constraints, 8)
                    .unwrap();

            let depth = count_depth(array.as_ref(), 1);

            println!(
                "Seed {}: depth={}, encoding={}",
                seed_offset,
                depth,
                array.encoding_id()
            );

            if depth > 1 {
                println!("  Tree:\n{}", array.display_as(DisplayOptions::TreeDisplay));
            }
        }
    }

    #[test]
    fn test_deeply_nested_dict_arrays() {
        // Seed that tends to produce DictArrays
        let seed: Vec<u8> = (200..5200).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        let constraints = ArrayConstraints::default();

        // Generate with high depth limit
        let array =
            arbitrary_constrained_array_with_depth(&mut u, Some(8), &dtype, &constraints, 10)
                .unwrap();

        let depth = count_depth(array.as_ref(), 1);
        println!("Generated array with depth: {}", depth);
        println!("Tree:\n{}", array.display_as(DisplayOptions::TreeDisplay));

        // Should sometimes produce nested structures
        // (depends on random choices, so we just verify it runs)
        assert!(array.len() > 0);
    }
}
