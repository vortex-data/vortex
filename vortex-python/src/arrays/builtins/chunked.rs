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

use crate::arrays::PyArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;
use crate::dtype::PyDType;
use crate::error::PyVortexResult;

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
    /// The chunks of a chunked array in `chunks` become chunks of the result, so the result has
    /// one level of chunks. For example, the arrays that `read_all` gives for several files
    /// combine into one array, and one `take` then reads rows of all the files.
    ///
    /// Parameters
    /// ----------
    /// chunks: list[vortex.Array]
    ///     The arrays to combine, in order. They must all have the same data type.
    /// dtype: vortex.DType | None
    ///     The data type of the result. It is necessary only when `chunks` is empty.
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
            (None, Some(first)) => first.dtype().clone(),
            (None, None) => {
                return Err(PyValueError::new_err(
                    "Cannot infer the dtype of a chunked array with no chunks; pass dtype.",
                )
                .into());
            }
        };

        Ok(PyArrayRef::from(
            ChunkedArray::try_new(flat, dtype)?.into_array(),
        ))
    }
}
