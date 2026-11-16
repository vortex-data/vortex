// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.box` extension type (`geoarrow.box`).

use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;

use super::SESSION;
use crate::test_harness::rect_column;

/// The existing spatial scalar functions run on a `Rect` operand via the shared `geometries()` decode,
/// producing the same results as the equivalent polygon: a box `(0,0)-(10,10)` against interior
/// point `(5,5)` and exterior point `(20,20)`.
#[test]
fn scalar_functions_run_on_rect() -> VortexResult<()> {
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;

    use crate::scalar_fn::contains::SpatialContains;
    use crate::scalar_fn::distance::SpatialDistance;
    use crate::scalar_fn::intersects::SpatialIntersects;
    use crate::test_harness::point_column;

    let mut ctx = SESSION.create_execution_ctx();
    let bbox = rect_column(vec![(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)])?;
    let points = point_column(vec![5.0, 20.0], vec![5.0, 20.0])?;

    // Distance: 0 to the interior point, >0 to the exterior point.
    let distance = SpatialDistance::try_new(bbox.clone(), points.clone())?.into_array();
    let distance = distance.execute::<Canonical>(&mut ctx)?.into_primitive();
    let distances = distance.as_slice::<f64>();
    assert_eq!(distances[0], 0.0);
    assert!((distances[1] - 200.0f64.sqrt()).abs() < 1e-12);

    // Intersects / Contains: true for the interior point, false for the exterior one.
    let intersects = SpatialIntersects::try_new(bbox.clone(), points.clone())?.into_array();
    assert_arrays_eq!(intersects, BoolArray::from_iter([true, false]), &mut ctx);

    let contains = SpatialContains::try_new(bbox, points)?.into_array();
    assert_arrays_eq!(contains, BoolArray::from_iter([true, false]), &mut ctx);
    Ok(())
}
