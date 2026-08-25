// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What the geo scalar functions add to the row-function machinery: an element type that decodes a
//! native geometry column into `geo_types` geometries.

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::unstable::row::InputElement;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::extension::geometries;
use crate::extension::is_native_geometry;

/// The geometry a null row decodes to. Arbitrary: valid-row execution never reads null slots. A
/// point is the cheapest well-formed variant (no allocation).
fn placeholder_geometry() -> Geometry<f64> {
    Geometry::Point(geo_types::Point::new(0.0, 0.0))
}

/// Marker for native geometry input elements: accepts any native geometry column and presents each
/// row as a decoded `geo_types` geometry.
///
/// The two operands of a binary geo function need not share a geometry type, since distance,
/// containment and intersection across types are all meaningful, so this validates only that the
/// column is _some_ native geometry.
pub(crate) struct GeometryRow;

// SAFETY: The row-loop view is a geometry slice. Its `ViewLen` is the slice length, so every index
// below that length is valid for both checked and unchecked access.
unsafe impl InputElement for GeometryRow {
    type Column = Vec<Geometry<f64>>;
    type View<'a> = &'a [Geometry<f64>];
    type Elem<'a> = &'a Geometry<f64>;

    // A geometry row is decoded from its coordinate storage, which behind a null row holds arbitrary
    // coordinates that need not describe a well-formed geometry.
    const DENSE_SAFE: bool = false;
    // Decoding builds a geometry from stored coordinates, and a malformed one in a *valid* row is a
    // domain error rather than an infrastructural failure.
    const DECODE_INFALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            is_native_geometry(dtype),
            "spatial: operand {dtype} is not a native geometry type"
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        geometries(&array, ctx)
    }

    // Every native geometry type decodes null rows to a placeholder.
    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }

    /// Null rows decode to [`placeholder_geometry`], which the branch-and-skip row loop never
    /// reads. Null rows are filtered out before decoding, so their storage — which can hold
    /// arbitrary payloads — is never interpreted; the placeholders keep the column indexed by row.
    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        let valid = array.validity()?.execute_mask(array.len(), ctx)?;
        if valid.all_true() {
            return geometries(&array, ctx).map(Some);
        }

        let mut decoded = geometries(&array.filter(valid.clone())?, ctx)?.into_iter();
        valid
            .to_bit_buffer()
            .iter()
            .map(|is_valid| {
                if is_valid {
                    decoded.next().ok_or_else(|| {
                        vortex_err!("spatial: filtered geometry column dropped a valid row")
                    })
                } else {
                    Ok(placeholder_geometry())
                }
            })
            .collect::<VortexResult<Vec<_>>>()
            .map(Some)
    }

    fn get(column: &Self::Column, index: usize) -> &Geometry<f64> {
        &column[index]
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column.as_slice()
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> &'a Geometry<f64>
    where
        Self: 'a,
    {
        &view[index]
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> &'a Geometry<f64>
    where
        Self: 'a,
    {
        // SAFETY: The caller established that `index` is below `ViewLen::len` for this view, which
        // is the geometry slice length.
        unsafe { view.get_unchecked(index) }
    }
}
