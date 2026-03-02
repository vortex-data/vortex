//! Defines a wrapper around
//! [`NVENCSTATUS`](crate::sys::nvEncodeAPI::NVENCSTATUS) to provide ergonomic
//! error handling.

use std::{error::Error, ffi::CStr, fmt};

use super::{api::ENCODE_API, encoder::Encoder};
use crate::sys::nvEncodeAPI::NVENCSTATUS;

/// Wrapper enum around [`NVENCSTATUS`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorKind {
    /// No encode capable devices were detected.
    NoEncodeDevice = 1,
    /// The device passed by the client is not supported.
    UnsupportedDevice = 2,
    /// The encoder device supplied by the client is not valid.
    InvalidEncoderDevice = 3,
    /// The device passed to the API call is invalid.
    InvalidDevice = 4,
    /// The device passed to the API call is no longer available
    /// and needs to be reinitialized. The clients need to destroy the
    /// current encoder session by freeing the allocated input output
    /// buffers and destroying the device and create a new encoding session.
    DeviceNotExist = 5,
    /// One or more of the pointers passed to the API call is invalid.
    InvalidPtr = 6,
    /// The completion event passed in the [`EncodeAPI.encode_picture`]
    /// call is invalid.
    InvalidEvent = 7,
    /// One or more of the parameter passed to the API call is invalid.
    InvalidParam = 8,
    /// An API call was made in wrong sequence or order.
    InvalidCall = 9,
    /// the API call failed because it was unable to allocate enough memory
    /// to perform the requested operation.
    OutOfMemory = 10,
    /// The encoder has not been initialized with
    /// [`EncodeAPI.initialize_encoder`] or that initialization has failed.
    /// The client cannot allocate input or output buffers or do any encoding
    /// related operation before successfully initializing the encoder.
    EncoderNotInitialized = 11,
    /// An unsupported parameter was passed by the client.
    UnsupportedParam = 12,
    /// The [`EncodeAPI.lock_bitstream`] failed to lock the output
    /// buffer. This happens when the client makes a non-blocking lock call
    /// to access the output bitstream by passing the `doNotWait` flag.
    /// This is not a fatal error and client should retry the same operation
    /// after few milliseconds.
    LockBusy = 13,
    /// The size of the user buffer passed by the client is insufficient for
    /// the requested operation.
    NotEnoughBuffer = 14,
    /// An invalid struct version was used by the client.
    InvalidVersion = 15,
    /// [`EncodeAPI.map_input_resource`] failed to map the client provided
    /// input resource.
    MapFailed = 16,
    /// The encode driver requires more input buffers to produce an output
    /// bitstream. If this error is returned from [`EncodeAPI.encode_picture`],
    /// this is not a fatal error. If the client is encoding with B frames
    /// then, [`EncodeAPI.encode_picture`] might be buffering the input
    /// frame for re-ordering.
    ///
    /// A client operating in synchronous mode cannot call
    /// [`EncodeAPI.lock_bitstream`] on the output bitstream buffer if
    /// [`EncodeAPI.encode_picture`] returned this variant. The client must
    /// continue providing input frames until encode driver returns
    /// successfully. After a success the client
    /// can call [`EncodeAPI.lock_bitstream`] on the output buffers in the
    /// same order in which it has called [`EncodeAPI.encode_picture`].
    NeedMoreInput = 17,
    /// The hardware encoder is busy encoding and is unable to encode
    /// the input. The client should call [`EncodeAPI.encode_picture`] again
    /// after few milliseconds.
    EncoderBusy = 18,
    /// The completion event passed in [`EncodeAPI.encode_picture`]
    /// has not been registered with encoder driver using
    /// [`EncodeAPI.register_async_event`].
    EventNotRegistered = 19,
    /// An unknown internal error has occurred.
    Generic = 20,
    /// The client is attempting to use a feature
    /// that is not available for the license type for the current system.
    IncompatibleClientKey = 21,
    /// the client is attempting to use a feature
    /// that is not implemented for the current version.
    Unimplemented = 22,
    /// [`EncodeAPI.register_resource`] failed to register the resource.
    ResourceRegisterFailed = 23,
    /// The client is attempting to unregister a resource
    /// that has not been successfully registered.
    ResourceNotRegistered = 24,
    /// The client is attempting to unmap a resource
    /// that has not been successfully mapped.
    ResourceNotMapped = 25,
    /// The encode driver requires more output buffers to write an
    /// output bitstream. If this error is returned from
    /// [`EncodeAPI.restore_encoder_state`], this is not a fatal error. If the
    /// client is encoding with B frames then,
    /// [`EncodeAPI.restore_encoder_state`] API might be requiring the extra
    /// output buffer for accommodating overlay frame output in a separate
    /// buffer, for AV1 codec. In this case, the client must call
    /// [`EncodeAPI.restore_encoder_state`] API again with
    /// an output bitstream as input along with the parameters in the previous
    /// call. When operating in asynchronous mode of encoding, client must
    /// also specify the completion event.
    NeedMoreOutput = 26,
}

/// Wrapper struct around [`NVENCSTATUS`].
///
/// This struct also contains a string with additional info
/// when it is relevant and available.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct EncodeError {
    kind: ErrorKind,
    string: Option<String>,
}

impl EncodeError {
    /// Getter for the error kind.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Getter for the error string.
    #[must_use]
    pub fn string(&self) -> Option<&str> {
        self.string.as_deref()
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.string {
            Some(s) => write!(f, "{:?}: {s}", self.kind),
            None => write!(f, "{:?}", self.kind),
        }
    }
}

impl Error for EncodeError {}

impl From<NVENCSTATUS> for ErrorKind {
    fn from(status: NVENCSTATUS) -> Self {
        match status {
            NVENCSTATUS::NV_ENC_SUCCESS => {
                unreachable!("Success should not be converted to an error.")
            }
            NVENCSTATUS::NV_ENC_ERR_NO_ENCODE_DEVICE => Self::NoEncodeDevice,
            NVENCSTATUS::NV_ENC_ERR_UNSUPPORTED_DEVICE => Self::UnsupportedDevice,
            NVENCSTATUS::NV_ENC_ERR_INVALID_ENCODERDEVICE => Self::InvalidEncoderDevice,
            NVENCSTATUS::NV_ENC_ERR_INVALID_DEVICE => Self::InvalidDevice,
            NVENCSTATUS::NV_ENC_ERR_DEVICE_NOT_EXIST => Self::DeviceNotExist,
            NVENCSTATUS::NV_ENC_ERR_INVALID_PTR => Self::InvalidPtr,
            NVENCSTATUS::NV_ENC_ERR_INVALID_EVENT => Self::InvalidEvent,
            NVENCSTATUS::NV_ENC_ERR_INVALID_PARAM => Self::InvalidParam,
            NVENCSTATUS::NV_ENC_ERR_INVALID_CALL => Self::InvalidCall,
            NVENCSTATUS::NV_ENC_ERR_OUT_OF_MEMORY => Self::OutOfMemory,
            NVENCSTATUS::NV_ENC_ERR_ENCODER_NOT_INITIALIZED => Self::EncoderNotInitialized,
            NVENCSTATUS::NV_ENC_ERR_UNSUPPORTED_PARAM => Self::UnsupportedParam,
            NVENCSTATUS::NV_ENC_ERR_LOCK_BUSY => Self::LockBusy,
            NVENCSTATUS::NV_ENC_ERR_NOT_ENOUGH_BUFFER => Self::NotEnoughBuffer,
            NVENCSTATUS::NV_ENC_ERR_INVALID_VERSION => Self::InvalidVersion,
            NVENCSTATUS::NV_ENC_ERR_MAP_FAILED => Self::MapFailed,
            NVENCSTATUS::NV_ENC_ERR_NEED_MORE_INPUT => Self::NeedMoreInput,
            NVENCSTATUS::NV_ENC_ERR_ENCODER_BUSY => Self::EncoderBusy,
            NVENCSTATUS::NV_ENC_ERR_EVENT_NOT_REGISTERD => Self::EventNotRegistered,
            NVENCSTATUS::NV_ENC_ERR_GENERIC => Self::Generic,
            NVENCSTATUS::NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY => Self::IncompatibleClientKey,
            NVENCSTATUS::NV_ENC_ERR_UNIMPLEMENTED => Self::Unimplemented,
            NVENCSTATUS::NV_ENC_ERR_RESOURCE_REGISTER_FAILED => Self::ResourceRegisterFailed,
            NVENCSTATUS::NV_ENC_ERR_RESOURCE_NOT_REGISTERED => Self::ResourceNotRegistered,
            NVENCSTATUS::NV_ENC_ERR_RESOURCE_NOT_MAPPED => Self::ResourceNotMapped,
            NVENCSTATUS::NV_ENC_ERR_NEED_MORE_OUTPUT => Self::NeedMoreOutput,
        }
    }
}

impl NVENCSTATUS {
    /// Convert an [`NVENCSTATUS`] to a [`Result`].
    ///
    /// [`NVENCSTATUS::NV_ENC_SUCCESS`] is converted to `Ok(())`,
    /// and all other variants are mapped to the corresponding variant
    /// in [`ErrorKind`]. The error type is [`EncodeError`] which has
    /// a kind and an optional `String` which might contain additional
    /// information about the error.
    ///
    /// # Errors
    ///
    /// Returns an error whenever the status is not
    /// [`NVENCSTATUS::NV_ENC_SUCCESS`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use cudarc::driver::CudaContext;
    /// # use nvidia_video_codec_sdk::{sys::nvEncodeAPI::GUID, EncodeError, Encoder, ErrorKind};
    /// # let cuda_ctx = CudaContext::new(0).unwrap();
    /// let encoder = Encoder::initialize_with_cuda(cuda_ctx).unwrap();
    /// // Cause an error by passing in an invalid GUID.
    /// // `Encoder::get_supported_input_formats()` uses `.result()` internally
    /// let error = encoder
    ///     .get_supported_input_formats(GUID::default())
    ///     .unwrap_err();
    /// // Get the kind.
    /// assert_eq!(error.kind(), ErrorKind::InvalidParam);
    /// // Get the error message.
    /// // Unfortunately, it's not always helpful.
    /// assert_eq!(error.string(), Some("EncodeAPI Internal Error."));
    /// ```
    pub fn result(self, encoder: &Encoder) -> Result<(), EncodeError> {
        self.result_without_string().map_err(|mut err| {
            err.string = match err.kind {
                // Avoid getting the string if it is not needed.
                ErrorKind::LockBusy
                | ErrorKind::EncoderBusy
                | ErrorKind::NeedMoreInput
                | ErrorKind::OutOfMemory => None,
                // Otherwise allocate an owned `String` with the error.
                _ => Some(
                    unsafe { CStr::from_ptr((ENCODE_API.get_last_error_string)(encoder.ptr)) }
                        .to_string_lossy()
                        .to_string(),
                ),
            }
            .and_then(|s| if s.is_empty() { None } else { Some(s) });
            err
        })
    }

    /// Convert an [`NVENCSTATUS`] to a [`Result`] without
    /// using an [`Encoder`].
    ///
    /// This function is the same as [`NVENCSTATUS::result`] except
    /// it does not get the error string because it does not have access
    /// to an [`Encoder`]. This is only useful if you do not have an [`Encoder`]
    /// yet, for example when initializing the API.
    ///
    /// You should always prefer to use [`NVENCSTATUS::result`] when possible.
    ///
    /// # Errors
    ///
    /// Returns an error whenever the status is not
    /// [`NVENCSTATUS::NV_ENC_SUCCESS`].
    pub fn result_without_string(self) -> Result<(), EncodeError> {
        match self {
            Self::NV_ENC_SUCCESS => Ok(()),
            err => Err(EncodeError {
                kind: err.into(),
                string: None,
            }),
        }
    }
}
