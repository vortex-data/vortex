// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `hilbert`: a locality-preserving `u32` key for spatially clustering a native geometry column.
//!
//! The caller supplies one dataset-wide [`Rect`](crate::extension::Rect) bound. Each geometry is
//! represented by the center of its XY envelope, quantized to 16 bits per axis within that bound,
//! and encoded on a Hilbert curve. The scalar function only computes keys; sorting the complete
//! row set and writing it in key order is an ingestion-layer responsibility.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::Rect;
use crate::extension::coordinate::ordinates;
use crate::extension::is_native_geometry;
use crate::scalar_fn::envelope::SpatialEnvelope;
use crate::scalar_fn::execute::Execution;
use crate::scalar_fn::execute::Operand;
use crate::scalar_fn::execute::dispatch_unary;

const AXIS_MAX: f64 = u16::MAX as f64;

#[expect(
    clippy::cast_possible_truncation,
    reason = "Hilbert quantization intentionally truncates onto a 16-bit grid"
)]
fn quantize(value: f64, min: f64, max: f64) -> Option<u32> {
    if !value.is_finite() {
        return None;
    }
    if min == max {
        return Some(0);
    }
    let normalized = (value - min) / (max - min);
    normalized
        .is_finite()
        .then(|| (normalized * AXIS_MAX).clamp(0.0, AXIS_MAX) as u32)
}

fn hilbert_key([xmin, ymin, xmax, ymax]: [f64; 4], x: f64, y: f64) -> Option<u32> {
    Some(hilbert_encode_16(
        quantize(x, xmin, xmax)?,
        quantize(y, ymin, ymax)?,
    ))
}

/// Return the midpoint without overflowing when finite endpoints have opposite signs or are near
/// the edge of the `f64` range.
#[inline]
fn midpoint(min: f64, max: f64) -> f64 {
    min / 2.0 + max / 2.0
}

#[inline]
fn envelope_center([xmin, ymin, xmax, ymax]: [f64; 4]) -> (f64, f64) {
    (midpoint(xmin, xmax), midpoint(ymin, ymax))
}

/// Interleave the low 16 bits of `value` with zero bits.
#[inline]
fn hilbert_interleave(mut value: u32) -> u32 {
    value = (value | (value << 8)) & 0x00ff_00ff;
    value = (value | (value << 4)) & 0x0f0f_0f0f;
    value = (value | (value << 2)) & 0x3333_3333;
    (value | (value << 1)) & 0x5555_5555
}

/// Encode a 16-bit-per-axis point as a 32-bit Hilbert index.
///
/// This is the public-domain prefix-scan algorithm from
/// <https://github.com/rawrunprotected/hilbert_curves>.
#[inline]
fn hilbert_encode_16(x: u32, y: u32) -> u32 {
    debug_assert!(x <= u32::from(u16::MAX));
    debug_assert!(y <= u32::from(u16::MAX));

    let input_x = x;
    let input_y = y;
    let mut state_a = x ^ y;
    let mut state_b = 0xffff ^ state_a;
    let mut state_c = 0xffff ^ (x | y);
    let mut state_d = x & (y ^ 0xffff);
    let mut next_a = state_a | (state_b >> 1);
    let mut next_b = (state_a >> 1) ^ state_a;
    let mut next_c = ((state_c >> 1) ^ (state_b & (state_d >> 1))) ^ state_c;
    let mut next_d = ((state_a & (state_c >> 1)) ^ (state_d >> 1)) ^ state_d;

    state_a = next_a;
    state_b = next_b;
    state_c = next_c;
    state_d = next_d;
    next_a = (state_a & (state_a >> 2)) ^ (state_b & (state_b >> 2));
    next_b = (state_a & (state_b >> 2)) ^ (state_b & ((state_a ^ state_b) >> 2));
    next_c ^= (state_a & (state_c >> 2)) ^ (state_b & (state_d >> 2));
    next_d ^= (state_b & (state_c >> 2)) ^ ((state_a ^ state_b) & (state_d >> 2));

    state_a = next_a;
    state_b = next_b;
    state_c = next_c;
    state_d = next_d;
    next_a = (state_a & (state_a >> 4)) ^ (state_b & (state_b >> 4));
    next_b = (state_a & (state_b >> 4)) ^ (state_b & ((state_a ^ state_b) >> 4));
    next_c ^= (state_a & (state_c >> 4)) ^ (state_b & (state_d >> 4));
    next_d ^= (state_b & (state_c >> 4)) ^ ((state_a ^ state_b) & (state_d >> 4));

    state_a = next_a;
    state_b = next_b;
    state_c = next_c;
    state_d = next_d;
    next_c ^= (state_a & (state_c >> 8)) ^ (state_b & (state_d >> 8));
    next_d ^= (state_b & (state_c >> 8)) ^ ((state_a ^ state_b) & (state_d >> 8));

    let state_a = next_c ^ (next_c >> 1);
    let state_b = next_d ^ (next_d >> 1);
    let i0 = input_x ^ input_y;
    let i1 = state_b | (0xffff ^ (i0 | state_a));

    (hilbert_interleave(i1) << 1) | hilbert_interleave(i0)
}

/// Read the four coordinates from the constant [`Rect`] bounds.
fn rect_bounds(scalar: &Scalar) -> VortexResult<[f64; 4]> {
    let storage = scalar.as_extension().to_storage_scalar();
    let fields = storage.as_struct();
    let read = |name: &str| -> VortexResult<f64> {
        f64::try_from(
            &fields
                .field(name)
                .ok_or_else(|| vortex_err!("geo: hilbert bounds missing {name}"))?,
        )
    };
    let bounds = [read("xmin")?, read("ymin")?, read("xmax")?, read("ymax")?];
    let [xmin, ymin, xmax, ymax] = bounds;
    vortex_ensure!(
        bounds.into_iter().all(f64::is_finite),
        "geo: hilbert bounds must be finite"
    );
    vortex_ensure!(
        xmin <= xmax && ymin <= ymax,
        "geo: hilbert bounds must satisfy xmin <= xmax and ymin <= ymax"
    );
    Ok(bounds)
}

fn geometry_keys(
    geometry: ArrayRef,
    bounds: [f64; 4],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let envelopes = SpatialEnvelope::try_new_array(geometry)?
        .into_array()
        .execute::<ExtensionArray>(ctx)?;
    let storage = envelopes.storage_array().clone();
    let valid = storage.validity()?.execute_mask(storage.len(), ctx)?;
    let boxes = storage.execute::<StructArray>(ctx)?;
    let xmins = ordinates(&boxes, "xmin", ctx)?;
    let ymins = ordinates(&boxes, "ymin", ctx)?;
    let xmaxs = ordinates(&boxes, "xmax", ctx)?;
    let ymaxs = ordinates(&boxes, "ymax", ctx)?;
    let mut keys = BufferMut::zeroed(xmins.len());
    let mut key_valid = vec![false; xmins.len()];
    for row in 0..keys.len() {
        let (x, y) = envelope_center([xmins[row], ymins[row], xmaxs[row], ymaxs[row]]);
        if let Some(key) = hilbert_key(bounds, x, y) {
            keys[row] = key;
            key_valid[row] = true;
        }
    }
    let key_valid = Mask::from(BitBuffer::from_iter(key_valid));
    let validity = Validity::from_mask(&valid & &key_valid, Nullability::Nullable);
    Ok(PrimitiveArray::new(keys.freeze(), validity).into_array())
}

fn execute_hilbert(
    execution: Execution<1, Validity>,
    bounds: [f64; 4],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match execution.operands {
        [Operand::Constant(geometry)] => {
            let one = ConstantArray::new(geometry, 1).into_array();
            let key = geometry_keys(one, bounds, ctx)?.execute_scalar(0, ctx)?;
            Ok(ConstantArray::new(key, execution.len).into_array())
        }
        [Operand::Column(geometry)] => geometry_keys(geometry, bounds, ctx),
    }
}

fn validate_hilbert_operands(dtypes: &[DType]) -> VortexResult<()> {
    vortex_ensure!(
        dtypes.len() == 2,
        "geo: hilbert requires a geometry and one whole-column bound, got {} operands",
        dtypes.len()
    );
    vortex_ensure!(
        is_native_geometry(&dtypes[0]),
        "geo: hilbert operand {} is not a native geometry type",
        dtypes[0]
    );
    vortex_ensure!(
        dtypes[1]
            .as_extension_opt()
            .is_some_and(|ext| ext.is::<Rect>()),
        "geo: hilbert bounds {} are not a geometry box",
        dtypes[1]
    );
    Ok(())
}

/// A locality-preserving `u32` key for the center of each geometry's XY envelope.
///
/// One common bounds scalar must cover the complete column being clustered. Null, empty, or
/// non-finite geometries yield null keys. This function computes keys only; it does not sort rows.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialHilbert;

impl SpatialHilbert {
    /// A lazy Hilbert-key column using one dataset-wide `bounds` scalar for every geometry row.
    pub fn try_new_array(geometry: ArrayRef, bounds: Scalar) -> VortexResult<ScalarFnArray> {
        let len = geometry.len();
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialHilbert, EmptyOptions).erased(),
            vec![geometry, ConstantArray::new(bounds, len).into_array()],
        )
    }
}

impl ScalarFnVTable for SpatialHilbert {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.hilbert");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("geometry"),
            1 => ChildName::from("bounds"),
            _ => unreachable!("hilbert has exactly two children"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        validate_hilbert_operands(dtypes)?;
        Ok(DType::Primitive(PType::U32, Nullability::Nullable))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let geometry = args.get(0)?;
        let bounds = args.get(1)?;
        let bounds = bounds.as_::<Constant>();
        if bounds.scalar().is_null() {
            return Ok(ConstantArray::new(
                Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable)),
                geometry.len(),
            )
            .into_array());
        }
        let bounds = rect_bounds(bounds.scalar())?;
        dispatch_unary(
            &geometry,
            DType::Primitive(PType::U32, Nullability::Nullable),
            |execution, ctx| execute_hilbert(execution, bounds, ctx),
            ctx,
        )
    }

    fn validity(&self, _: &Self::Options, _: &Expression) -> VortexResult<Option<Expression>> {
        // Empty and non-finite geometries yield null even when both inputs are valid.
        Ok(None)
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::SpatialHilbert;
    use super::hilbert_encode_16;
    use crate::aggregate_fn::GeometryAabb;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::rect_column;

    fn bounds(
        corners: (f64, f64, f64, f64),
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<Scalar> {
        rect_column(vec![corners])?.execute_scalar(0, ctx)
    }

    fn keys(geometry: ArrayRef, bounds: Scalar) -> VortexResult<ArrayRef> {
        Ok(SpatialHilbert::try_new_array(geometry, bounds)?.into_array())
    }

    /// Reference values lock down scaling, truncation, orientation, and bit ordering.
    #[test]
    fn matches_reference_vector() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let points = point_column(vec![0.25, 0.50, 0.75], vec![0.25, 0.50, 0.75])?;
        let expected = PrimitiveArray::new(
            vec![178_956_970_u32, 715_827_882, 2_326_440_618],
            Validity::from_iter([true, true, true]),
        )
        .into_array();

        assert_arrays_eq!(
            keys(points, bounds((0.0, 0.0, 1.0, 1.0), &mut ctx)?)?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    /// A point and geometries whose envelopes have the same center receive the same key under one
    /// common domain.
    #[test]
    fn keys_the_envelope_center() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let bounds = bounds((0.0, 0.0, 10.0, 10.0), &mut ctx)?;
        let point = point_column(vec![2.0], vec![3.0])?;
        let multipoint = multipoint_column(vec![vec![(0.0, 1.0), (4.0, 5.0)]])?;
        let rect = rect_column(vec![(0.0, 1.0, 4.0, 5.0)])?;
        let expected = PrimitiveArray::new(
            vec![hilbert_encode_16(13_107, 19_660)],
            Validity::from_iter([true]),
        )
        .into_array();

        for geometry in [point, multipoint, rect] {
            assert_arrays_eq!(keys(geometry, bounds.clone())?, expected.clone(), &mut ctx);
        }
        Ok(())
    }

    /// The existing whole-column AABB aggregate produces exactly the scalar accepted by Hilbert.
    #[test]
    fn consumes_geometry_aabb_result() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let points = point_column(vec![0.0, 0.5, 1.0], vec![0.0, 0.5, 1.0])?;
        let mut aabb =
            Accumulator::try_new(GeometryAabb, AggregateEmptyOptions, points.dtype().clone())?;
        aabb.accumulate(&points, &mut ctx)?;

        let expected = PrimitiveArray::new(
            vec![0_u32, 715_827_882, 2_863_311_530],
            Validity::from_iter([true, true, true]),
        )
        .into_array();
        assert_arrays_eq!(keys(points, aabb.finish()?)?, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn constant_geometry_is_broadcast() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let point = point_column(vec![1.0], vec![1.0])?.execute_scalar(0, &mut ctx)?;
        let geometries = ConstantArray::new(point, 3).into_array();
        let expected = PrimitiveArray::new(
            vec![715_827_882_u32; 3],
            Validity::from_iter([true, true, true]),
        )
        .into_array();

        assert_arrays_eq!(
            keys(geometries, bounds((0.0, 0.0, 2.0, 2.0), &mut ctx)?)?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    /// Null and empty geometries retain their row positions as null keys.
    #[test]
    fn empty_and_null_geometries_are_null() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let geometries = nullable_multipolygon_column(vec![
            Some(vec![vec![vec![(0.0, 0.0), (2.0, 2.0)]]]),
            Some(vec![]),
            None,
        ])?;
        let expected = PrimitiveArray::new(
            vec![715_827_882_u32, 0, 0],
            Validity::from_iter([true, false, false]),
        )
        .into_array();

        assert_arrays_eq!(
            keys(geometries, bounds((0.0, 0.0, 2.0, 2.0), &mut ctx)?)?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    /// Non-finite point coordinates do not become arbitrary sortable keys.
    #[test]
    fn non_finite_geometry_is_null() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let points = point_column(vec![f64::NAN, 1.0], vec![f64::NAN, 1.0])?;
        let expected =
            PrimitiveArray::new(vec![0_u32, 715_827_882], Validity::from_iter([false, true]))
                .into_array();

        assert_arrays_eq!(
            keys(points, bounds((0.0, 0.0, 2.0, 2.0), &mut ctx)?)?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    /// A zero-width dataset axis is valid: it contributes no ordering information while the
    /// other axis remains sortable.
    #[test]
    fn degenerate_axis_is_supported() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let points = point_column(vec![1.0, 1.0], vec![0.0, 2.0])?;
        let expected = PrimitiveArray::new(
            vec![0_u32, 1_431_655_765],
            Validity::from_iter([true, true]),
        )
        .into_array();

        assert_arrays_eq!(
            keys(points, bounds((1.0, 0.0, 1.0, 2.0), &mut ctx)?)?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    /// Slicing still attributes nested coordinates to the correct outer geometry row.
    #[test]
    fn sliced_nested_geometry_keeps_rows_aligned() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let geometries = multipolygon_column(vec![
            vec![vec![vec![(-100.0, -100.0), (100.0, 100.0)]]],
            vec![vec![vec![(0.0, 0.0), (2.0, 2.0)]]],
            vec![vec![vec![(4.0, 4.0), (6.0, 6.0)]]],
        ])?;
        let expected = PrimitiveArray::new(
            vec![
                hilbert_encode_16(8_191, 8_191),
                hilbert_encode_16(40_959, 40_959),
            ],
            Validity::from_iter([true, true]),
        )
        .into_array();

        assert_arrays_eq!(
            keys(
                geometries.slice(1..3)?,
                bounds((0.0, 0.0, 8.0, 8.0), &mut ctx)?
            )?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    /// Planning rejects non-geometry operands and non-box bounds before execution.
    #[test]
    fn validates_operand_types() -> VortexResult<()> {
        let numeric = DType::Primitive(PType::I32, Nullability::NonNullable);
        let geometry = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        assert!(
            SpatialHilbert
                .return_dtype(&EmptyOptions, &[numeric.clone(), geometry.clone()])
                .is_err()
        );
        assert!(
            SpatialHilbert
                .return_dtype(&EmptyOptions, &[geometry, numeric])
                .is_err()
        );
        Ok(())
    }

    /// Execution materializes a canonical nullable `u32` primitive array.
    #[test]
    fn result_is_nullable_u32() -> VortexResult<()> {
        let session = crate::test_harness::spatial_session();
        let mut ctx = session.create_execution_ctx();
        let result = keys(
            nullable_point_column(vec![Some((1.0, 1.0)), None])?,
            bounds((0.0, 0.0, 2.0, 2.0), &mut ctx)?,
        )?
        .execute::<Canonical>(&mut ctx)?
        .into_primitive();
        assert_eq!(result.ptype(), PType::U32);
        assert!(result.dtype().is_nullable());
        Ok(())
    }
}
