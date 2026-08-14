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

use crate::extension::can_decode_geometries_null_tolerant;
use crate::extension::geometries;
use crate::extension::geometries_null_tolerant;
use crate::extension::is_native_geometry;

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

    // A geometry row is decoded from its coordinate storage, which behind a null row holds
    // arbitrary coordinates that need not describe a well-formed geometry.
    const DENSE_SAFE: bool = false;
    // Decoding builds a geometry from stored coordinates, and a malformed one in a _valid_ row is a
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

    fn can_decode_null_tolerant(array: &ArrayRef) -> VortexResult<bool> {
        can_decode_geometries_null_tolerant(array)
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

    /// Null rows decode to a placeholder geometry that the branch-and-skip row loop never reads.
    /// `Point` and `Polygon` columns are covered; other geometry types return `Ok(None)` and the
    /// batch falls back to the filter strategy.
    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        geometries_null_tolerant(&array, ctx)
    }
}

/// Test-only support for the prepared geo row kernels: a probe recording which operands a
/// `prepare` step saw as batch-constant, and the shared prepared-vs-expanded agreement check
/// built on it.
#[cfg(test)]
pub(crate) mod probe {
    use std::cell::Cell;

    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    thread_local! {
        /// Which operands the last `prepare` saw as constant, as a bitmask (bit 0 for `a`, bit 1
        /// for `b`). Thread-local rather than a process global so concurrent tests in one process
        /// (plain `cargo test`) cannot race it; execution runs on the calling thread.
        pub(crate) static SEEN_CONSTANTS: Cell<u8> = const { Cell::new(u8::MAX) };
    }

    /// Record which operands `prepare` saw as constant.
    pub(crate) fn record(a_constant: bool, b_constant: bool) {
        SEEN_CONSTANTS.set(u8::from(a_constant) | (u8::from(b_constant) << 1));
    }

    /// Execute `build(a, b)` and assert that `prepare` saw exactly `expect_seen` as its constant
    /// operands, so the test knows which decode path the inputs took.
    fn run_probed(
        build: &impl Fn(ArrayRef, ArrayRef) -> VortexResult<ScalarFnArray>,
        a: ArrayRef,
        b: ArrayRef,
        expect_seen: u8,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        SEEN_CONSTANTS.set(u8::MAX);
        let result = build(a, b)?
            .into_array()
            .execute::<Canonical>(ctx)?
            .into_array();

        assert_eq!(
            SEEN_CONSTANTS.get(),
            expect_seen,
            "prepare saw the wrong constant operands",
        );
        Ok(result)
    }

    /// Assert that every constant-operand arrangement of `build(a, b)` returns exactly what the
    /// fully expanded columns return, and that each arrangement's constness really reached
    /// `prepare` (so the constants exercised the stride-0 path rather than a decoded column).
    ///
    /// Arrangements: `a` constant, `b` constant, and both constant with `a` masked. A plain
    /// constant pair folds to a single-row execution before the row loop, so masking one side is
    /// what drives the both-hoisted arm across rows; that run is compared against the same mask
    /// over the expanded column.
    pub(crate) fn assert_prepared_agrees_with_columns(
        build: impl Fn(ArrayRef, ArrayRef) -> VortexResult<ScalarFnArray>,
        const_a: ArrayRef,
        const_b: ArrayRef,
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let column_a = const_a.clone().execute::<Canonical>(&mut ctx)?.into_array();
        let column_b = const_b.clone().execute::<Canonical>(&mut ctx)?.into_array();

        let baseline = run_probed(&build, column_a.clone(), column_b.clone(), 0b00, &mut ctx)?;
        let a_hoisted = run_probed(&build, const_a.clone(), column_b.clone(), 0b01, &mut ctx)?;
        let b_hoisted = run_probed(&build, column_a.clone(), const_b.clone(), 0b10, &mut ctx)?;
        assert_arrays_eq!(a_hoisted, baseline, &mut ctx);
        assert_arrays_eq!(b_hoisted, baseline, &mut ctx);

        let validity = Validity::from_iter((0..column_a.len()).map(|row| row != 1));
        let masked_const_a = MaskedArray::try_new(const_a, validity.clone())?.into_array();
        let masked_column_a = MaskedArray::try_new(column_a, validity)?.into_array();
        let both_hoisted = run_probed(&build, masked_const_a, const_b, 0b11, &mut ctx)?;
        let masked_baseline = run_probed(&build, masked_column_a, column_b, 0b00, &mut ctx)?;
        assert_arrays_eq!(both_hoisted, masked_baseline, &mut ctx);

        Ok(())
    }
}
