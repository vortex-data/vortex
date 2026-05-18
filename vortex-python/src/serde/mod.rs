// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod context;
pub(crate) mod parts;

use bytes::Bytes;
use pyo3::Bound;
use pyo3::Python;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::dtype::DType;
use vortex::ipc::messages::DecoderMessage;
use vortex::ipc::messages::EncoderMessage;
use vortex::ipc::messages::MessageDecoder;
use vortex::ipc::messages::MessageEncoder;
use vortex::ipc::messages::PollRead;
use vortex::session::VortexSession;

use crate::arrays::PyArrayRef;
use crate::error::PyVortexResult;
use crate::install_module;
use crate::serde::context::PyArrayContext;
use crate::serde::context::PyReadContext;
use crate::serde::parts::PySerializedArray;
use crate::session::session;

type PyIpcArrayBuffers = (Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Register serde functions and classes.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "serde")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.serde", &m)?;

    m.add_class::<PySerializedArray>()?;
    m.add_class::<PyArrayContext>()?;
    m.add_class::<PyReadContext>()?;
    m.add_function(wrap_pyfunction!(encode_ipc_array_buffers, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_ipc_array, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_ipc_array_buffers, &m)?)?;

    Ok(())
}

/// Encode a Vortex array into IPC array and dtype buffers.
#[pyfunction]
fn encode_ipc_array_buffers(py: Python, array: PyArrayRef) -> PyVortexResult<PyIpcArrayBuffers> {
    let session = session();
    let array = array.into_inner();
    py.detach(move || {
        let mut encoder = MessageEncoder::new(session.clone());
        let array_buffers = encoder
            .encode(EncoderMessage::Array(&array))?
            .iter()
            .map(|buffer| buffer.to_vec())
            .collect();
        let dtype_buffers = encoder
            .encode(EncoderMessage::DType(array.dtype()))?
            .iter()
            .map(|buffer| buffer.to_vec())
            .collect();
        Ok((array_buffers, dtype_buffers))
    })
}

/// Decode a Vortex array from IPC-encoded bytes.
///
/// This function decodes both the dtype and array messages from IPC format
/// and returns the reconstructed array.
///
/// Parameters
/// ----------
/// array_bytes : bytes
///     The IPC-encoded array message
/// dtype_bytes : bytes
///     The IPC-encoded dtype message
///
/// Returns
/// -------
/// Array
///     The decoded Vortex array
#[pyfunction]
#[pyo3(signature = (array_bytes, dtype_bytes))]
fn decode_ipc_array(
    py: Python,
    array_bytes: Vec<u8>,
    dtype_bytes: Vec<u8>,
) -> PyVortexResult<PyArrayRef> {
    let session = session();
    let array =
        py.detach(move || decode_ipc_array_from_bytes(array_bytes, dtype_bytes, session))?;
    Ok(PyArrayRef::from(array))
}

fn decode_ipc_array_from_bytes(
    array_bytes: Vec<u8>,
    dtype_bytes: Vec<u8>,
    session: &VortexSession,
) -> PyVortexResult<ArrayRef> {
    let mut decoder = MessageDecoder::default();

    let mut dtype_buf = Bytes::from(dtype_bytes);
    let dtype = match decoder.read_next(&mut dtype_buf)? {
        PollRead::Some(DecoderMessage::DType(dtype)) => dtype,
        PollRead::Some(_) => {
            return Err(PyValueError::new_err("Expected DType message").into());
        }
        PollRead::NeedMore(_) => {
            return Err(PyValueError::new_err("Incomplete DType message").into());
        }
    };
    let dtype = DType::from_flatbuffer(dtype, session)?;

    let mut array_buf = Bytes::from(array_bytes);
    let array = match decoder.read_next(&mut array_buf)? {
        PollRead::Some(DecoderMessage::Array((parts, ctx, row_count))) => {
            parts.decode(&dtype, row_count, &ctx, session)?
        }
        PollRead::Some(_) => {
            return Err(PyValueError::new_err("Expected Array message").into());
        }
        PollRead::NeedMore(_) => {
            return Err(PyValueError::new_err("Incomplete Array message").into());
        }
    };

    Ok(array)
}

/// Decode a Vortex array from IPC-encoded buffer protocol objects
///
/// This function accepts lists of buffer protocol objects (memoryviews) and decodes
/// them without copying by using PyO3's buffer protocol support.
///
/// Parameters
/// ----------
/// array_buffers : list of buffer protocol objects
///     List of IPC-encoded array message buffers
/// dtype_buffers : list of buffer protocol objects
///     List of IPC-encoded dtype message buffers
///
/// Returns
/// -------
/// Array
///     The decoded Vortex array
#[pyfunction]
#[pyo3(signature = (array_buffers, dtype_buffers))]
fn decode_ipc_array_buffers<'py>(
    py: Python<'py>,
    array_buffers: Vec<Bound<'py, PyAny>>,
    dtype_buffers: Vec<Bound<'py, PyAny>>,
) -> PyVortexResult<PyArrayRef> {
    // Concatenate dtype buffers
    // Note: PyBuffer returns &[ReadOnlyCell<u8>] which requires copying to get &[u8]
    let mut dtype_bytes_vec = Vec::new();
    for buf_obj in dtype_buffers {
        let buffer = PyBuffer::<u8>::get(&buf_obj)?;
        let slice = buffer
            .as_slice(py)
            .ok_or_else(|| PyValueError::new_err("Buffer is not contiguous"))?;
        for cell in slice {
            dtype_bytes_vec.push(cell.get());
        }
    }
    // Concatenate array buffers
    let mut array_bytes_vec = Vec::new();
    for buf_obj in array_buffers {
        let buffer = PyBuffer::<u8>::get(&buf_obj)?;
        let slice = buffer
            .as_slice(py)
            .ok_or_else(|| PyValueError::new_err("Buffer is not contiguous"))?;
        for cell in slice {
            array_bytes_vec.push(cell.get());
        }
    }

    let session = session();
    let array =
        py.detach(move || decode_ipc_array_from_bytes(array_bytes_vec, dtype_bytes_vec, session))?;
    Ok(PyArrayRef::from(array))
}
