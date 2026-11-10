// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated geometry families and metadata for Arrow transport checks.
//!
//! Each case samples one family so shrinking can reduce the geometry as well as its storage.

mod tests {
    use std::sync::Arc;

    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::builders::builder_with_capacity_in;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::Scalar;
    use vortex_arrow::test_harness::DefaultExport;
    use vortex_arrow::test_harness::check_generated;
    use vortex_arrow::test_harness::draw_storage;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::extension::LineString;
    use crate::extension::MultiLineString;
    use crate::extension::MultiPoint;
    use crate::extension::MultiPolygon;
    use crate::extension::Point;
    use crate::extension::Polygon;
    use crate::extension::Rect;
    use crate::extension::SpatialMetadata;
    use crate::extension::WellKnownBinary;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_storage_dtype;
    use crate::extension::rect::box_storage_dtype;

    fn check_spatial_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullability = tc.draw(gs::booleans()).into();
        let dimension = tc.draw(gs::sampled_from(&[
            Dimension::Xy,
            Dimension::Xyz,
            Dimension::Xym,
            Dimension::Xyzm,
        ]));
        let crs = tc.draw(gs::sampled_from(&[
            None,
            Some(""),
            Some("EPSG:4326"),
            Some("非ASCII"),
        ]));
        let metadata = SpatialMetadata::new(crs.map(Into::into));
        let coordinates = coordinate_storage_dtype(dimension, Nullability::NonNullable);
        let line = DType::List(Arc::new(coordinates.clone()), Nullability::NonNullable);
        let polygon = DType::List(Arc::new(line.clone()), Nullability::NonNullable);

        let dtypes = [
            ExtDType::try_with_vtable(
                Point,
                metadata.clone(),
                coordinates.with_nullability(nullability),
            )?
            .erased(),
            ExtDType::try_with_vtable(
                Rect,
                metadata.clone(),
                box_storage_dtype(dimension, nullability),
            )?
            .erased(),
            ExtDType::try_with_vtable(
                LineString,
                metadata.clone(),
                line.clone().with_nullability(nullability),
            )?
            .erased(),
            ExtDType::try_with_vtable(
                MultiPoint,
                metadata.clone(),
                line.with_nullability(nullability),
            )?
            .erased(),
            ExtDType::try_with_vtable(
                Polygon,
                metadata.clone(),
                polygon.clone().with_nullability(nullability),
            )?
            .erased(),
            ExtDType::try_with_vtable(
                MultiLineString,
                metadata.clone(),
                polygon.clone().with_nullability(nullability),
            )?
            .erased(),
            ExtDType::try_with_vtable(
                MultiPolygon,
                metadata.clone(),
                DType::List(Arc::new(polygon), nullability),
            )?
            .erased(),
            ExtDType::try_with_vtable(WellKnownBinary, metadata, DType::Binary(nullability))?
                .erased(),
        ];
        let dtype = tc.draw(gs::sampled_from(&dtypes));
        let storage = if dtype.is::<WellKnownBinary>() {
            draw_wkb_storage(tc, nullability, len)?
        } else {
            draw_storage(tc, dtype.storage_dtype(), len, &mut ctx)?
        };
        let array = ExtensionArray::try_new(dtype, storage)?.into_array();

        check_generated(tc, array, DefaultExport::Supported, &mut ctx)
    }

    /// Draws nullable storage containing a valid little-endian, two-dimensional WKB point.
    fn draw_wkb_storage(
        tc: &TestCase,
        nullability: Nullability,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let mut bytes = vec![1u8];
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&f64::from(tc.draw(gs::integers::<i32>())).to_le_bytes());
        bytes.extend_from_slice(&f64::from(tc.draw(gs::integers::<i32>())).to_le_bytes());
        let mut builder = builder_with_capacity_in(
            &DType::Binary(nullability),
            len,
            BufferAllocatorRef::static_ref(),
        );

        for _ in 0..len {
            if nullability.is_nullable() && tc.draw(gs::booleans()) {
                builder.append_nulls(1);
            } else {
                builder.append_scalar(&Scalar::binary(bytes.clone(), nullability))?;
            }
        }

        Ok(builder.finish())
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn spatial_arrow_transport(tc: TestCase) {
        check_spatial_round_trip(&tc)
            .vortex_expect("spatial Arrow transport must preserve dtype and stored values");
    }
}
