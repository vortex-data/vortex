// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::Bound;
use pyo3::PyRef;
use pyo3::exceptions::PyValueError;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::dtype::DType;
use vortex::error::VortexResult;

use crate::arrays::PyArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;
use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::session::session;

/// Concrete class for arrays with `vortex.chunked` encoding.
#[pyclass(name = "ChunkedArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyChunkedArray;

impl EncodingSubclass for PyChunkedArray {
    type VTable = Chunked;
}

#[pymethods]
impl PyChunkedArray {
    pub fn chunks(self_: PyRef<'_, Self>) -> Vec<PyArrayRef> {
        self_
            .as_super()
            .inner()
            .as_::<Chunked>()
            .iter_chunks()
            .map(|chunk| PyArrayRef::from(chunk.clone()))
            .collect()
    }

    /// Combine arrays of one data type into a chunked array, with no copy.
    ///
    /// Use it for arrays that are already in memory, such as the results of earlier reads or of
    /// several sources. One `take` on the result then reads rows of all the arrays, in any order.
    ///
    /// The chunks of a chunked array in `chunks` become chunks of the result, so the result has
    /// one level of chunks.
    ///
    /// Parameters
    /// ----------
    /// chunks: list[vortex.Array]
    ///     The arrays to combine, in order. They must all have the same data type, but they can
    ///     differ in nullability. If one of them is nullable, the result is nullable.
    /// dtype: vortex.DType | None
    ///     The data type of the result. It is necessary only when `chunks` is empty. A
    ///     non-nullable chunk can go into a nullable `dtype`.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.ChunkedArray`
    ///
    /// Examples
    /// --------
    ///
    /// ```python
    /// >>> a = vx.array([1, 2, 3])
    /// >>> b = vx.array([4, 5])
    /// >>> chunked = vx.ChunkedArray.from_chunks([a, b])
    /// >>> chunked.take(vx.array([4, 0])).to_arrow_array()
    /// <pyarrow.lib.Int64Array object at ...>
    /// [
    ///   5,
    ///   1
    /// ]
    /// ```
    #[staticmethod]
    #[pyo3(signature = (chunks, *, dtype = None))]
    fn from_chunks(
        chunks: Vec<PyArrayRef>,
        dtype: Option<Bound<PyDType>>,
    ) -> PyVortexResult<PyArrayRef> {
        let mut flat: Vec<ArrayRef> = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let chunk = chunk.into_inner();

            match chunk.as_opt::<Chunked>() {
                Some(chunked) => flat.extend(chunked.iter_chunks().cloned()),
                None => flat.push(chunk),
            }
        }

        let dtype = match (dtype, flat.first()) {
            (Some(dtype), _) => dtype.get().inner().clone(),

            // A chunk with nulls makes the result nullable.
            (None, Some(first)) if flat.iter().any(|chunk| chunk.dtype().is_nullable()) => {
                first.dtype().as_nullable()
            }

            (None, Some(first)) => first.dtype().clone(),

            (None, None) => {
                return Err(PyValueError::new_err(
                    "Cannot infer the dtype of a chunked array with no chunks; pass dtype.",
                )
                .into());
            }
        };

        let flat = flat
            .into_iter()
            .map(|chunk| as_nullable_chunk(chunk, &dtype))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(PyArrayRef::from(
            ChunkedArray::try_new(flat, dtype)?.into_array(),
        ))
    }
}

/// Cast a non-nullable chunk to `dtype` if `dtype` is the same type, but nullable.
///
/// The cast changes only the nullability, so it copies no data. Other chunks do not change, and
/// `ChunkedArray::try_new` rejects a chunk of a different type.
fn as_nullable_chunk(chunk: ArrayRef, dtype: &DType) -> VortexResult<ArrayRef> {
    if chunk.dtype().is_nullable() || &chunk.dtype().as_nullable() != dtype {
        return Ok(chunk);
    }

    chunk.cast(dtype.clone())?.optimize_ctx(session())
}
