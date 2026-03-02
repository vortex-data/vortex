// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Error types for NVENC operations.

use crate::sys;

/// Error type for NVENC operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvencError {
    /// Failed to load the NVENC library at runtime.
    LibraryLoadError(String),
    /// No encode-capable device found.
    NoEncodeDevice,
    /// Device is not supported for encoding.
    UnsupportedDevice,
    /// Invalid encoder device.
    InvalidEncoderDevice,
    /// Invalid device.
    InvalidDevice,
    /// Device does not exist.
    DeviceNotExist,
    /// Invalid pointer.
    InvalidPtr,
    /// Invalid event.
    InvalidEvent,
    /// Invalid parameter.
    InvalidParam,
    /// Invalid call.
    InvalidCall,
    /// Out of memory.
    OutOfMemory,
    /// Encoder not initialized.
    EncoderNotInitialized,
    /// Unsupported parameter.
    UnsupportedParam,
    /// Lock is busy.
    LockBusy,
    /// Not enough buffer.
    NotEnoughBuffer,
    /// Invalid version.
    InvalidVersion,
    /// Map failed.
    MapFailed,
    /// Need more input.
    NeedMoreInput,
    /// Encoder busy.
    EncoderBusy,
    /// Event not registered.
    EventNotRegistered,
    /// Generic error.
    Generic,
    /// Incompatible client key.
    IncompatibleClientKey,
    /// Unimplemented.
    Unimplemented,
    /// Resource register failed.
    ResourceRegisterFailed,
    /// Resource not registered.
    ResourceNotRegistered,
    /// Resource not mapped.
    ResourceNotMapped,
    /// Detailed error with context message.
    DetailedError(String),
    /// Unknown error with raw status code.
    Unknown(u32),
}

impl std::fmt::Display for NvencError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryLoadError(msg) => write!(f, "nvenc: failed to load library: {msg}"),
            Self::NoEncodeDevice => write!(f, "nvenc: no encode-capable device"),
            Self::UnsupportedDevice => write!(f, "nvenc: unsupported device"),
            Self::InvalidEncoderDevice => write!(f, "nvenc: invalid encoder device"),
            Self::InvalidDevice => write!(f, "nvenc: invalid device"),
            Self::DeviceNotExist => write!(f, "nvenc: device does not exist"),
            Self::InvalidPtr => write!(f, "nvenc: invalid pointer"),
            Self::InvalidEvent => write!(f, "nvenc: invalid event"),
            Self::InvalidParam => write!(f, "nvenc: invalid parameter"),
            Self::InvalidCall => write!(f, "nvenc: invalid call"),
            Self::OutOfMemory => write!(f, "nvenc: out of memory"),
            Self::EncoderNotInitialized => write!(f, "nvenc: encoder not initialized"),
            Self::UnsupportedParam => write!(f, "nvenc: unsupported parameter"),
            Self::LockBusy => write!(f, "nvenc: lock busy"),
            Self::NotEnoughBuffer => write!(f, "nvenc: not enough buffer"),
            Self::InvalidVersion => write!(f, "nvenc: invalid version"),
            Self::MapFailed => write!(f, "nvenc: map failed"),
            Self::NeedMoreInput => write!(f, "nvenc: need more input"),
            Self::EncoderBusy => write!(f, "nvenc: encoder busy"),
            Self::EventNotRegistered => write!(f, "nvenc: event not registered"),
            Self::Generic => write!(f, "nvenc: generic error"),
            Self::IncompatibleClientKey => write!(f, "nvenc: incompatible client key"),
            Self::Unimplemented => write!(f, "nvenc: unimplemented"),
            Self::ResourceRegisterFailed => write!(f, "nvenc: resource register failed"),
            Self::ResourceNotRegistered => write!(f, "nvenc: resource not registered"),
            Self::ResourceNotMapped => write!(f, "nvenc: resource not mapped"),
            Self::DetailedError(msg) => write!(f, "nvenc: {msg}"),
            Self::Unknown(code) => write!(f, "nvenc: unknown error (status code {code})"),
        }
    }
}

impl std::error::Error for NvencError {}

/// Checks an NVENC status code and converts it to a Result.
pub(crate) fn check_status(status: sys::NVENCSTATUS) -> Result<(), NvencError> {
    #![allow(non_upper_case_globals)]

    match status {
        sys::NVENCSTATUS_NV_ENC_SUCCESS => Ok(()),
        sys::NVENCSTATUS_NV_ENC_ERR_NO_ENCODE_DEVICE => Err(NvencError::NoEncodeDevice),
        sys::NVENCSTATUS_NV_ENC_ERR_UNSUPPORTED_DEVICE => Err(NvencError::UnsupportedDevice),
        sys::NVENCSTATUS_NV_ENC_ERR_INVALID_ENCODERDEVICE => Err(NvencError::InvalidEncoderDevice),
        sys::NVENCSTATUS_NV_ENC_ERR_INVALID_DEVICE => Err(NvencError::InvalidDevice),
        sys::NVENCSTATUS_NV_ENC_ERR_DEVICE_NOT_EXIST => Err(NvencError::DeviceNotExist),
        sys::NVENCSTATUS_NV_ENC_ERR_INVALID_PTR => Err(NvencError::InvalidPtr),
        sys::NVENCSTATUS_NV_ENC_ERR_INVALID_EVENT => Err(NvencError::InvalidEvent),
        sys::NVENCSTATUS_NV_ENC_ERR_INVALID_PARAM => Err(NvencError::InvalidParam),
        sys::NVENCSTATUS_NV_ENC_ERR_INVALID_CALL => Err(NvencError::InvalidCall),
        sys::NVENCSTATUS_NV_ENC_ERR_OUT_OF_MEMORY => Err(NvencError::OutOfMemory),
        sys::NVENCSTATUS_NV_ENC_ERR_ENCODER_NOT_INITIALIZED => {
            Err(NvencError::EncoderNotInitialized)
        }
        sys::NVENCSTATUS_NV_ENC_ERR_UNSUPPORTED_PARAM => Err(NvencError::UnsupportedParam),
        sys::NVENCSTATUS_NV_ENC_ERR_LOCK_BUSY => Err(NvencError::LockBusy),
        sys::NVENCSTATUS_NV_ENC_ERR_NOT_ENOUGH_BUFFER => Err(NvencError::NotEnoughBuffer),
        sys::NVENCSTATUS_NV_ENC_ERR_INVALID_VERSION => Err(NvencError::InvalidVersion),
        sys::NVENCSTATUS_NV_ENC_ERR_MAP_FAILED => Err(NvencError::MapFailed),
        sys::NVENCSTATUS_NV_ENC_ERR_NEED_MORE_INPUT => Err(NvencError::NeedMoreInput),
        sys::NVENCSTATUS_NV_ENC_ERR_ENCODER_BUSY => Err(NvencError::EncoderBusy),
        sys::NVENCSTATUS_NV_ENC_ERR_EVENT_NOT_REGISTERD => Err(NvencError::EventNotRegistered),
        sys::NVENCSTATUS_NV_ENC_ERR_GENERIC => Err(NvencError::Generic),
        sys::NVENCSTATUS_NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY => {
            Err(NvencError::IncompatibleClientKey)
        }
        sys::NVENCSTATUS_NV_ENC_ERR_UNIMPLEMENTED => Err(NvencError::Unimplemented),
        sys::NVENCSTATUS_NV_ENC_ERR_RESOURCE_REGISTER_FAILED => {
            Err(NvencError::ResourceRegisterFailed)
        }
        sys::NVENCSTATUS_NV_ENC_ERR_RESOURCE_NOT_REGISTERED => {
            Err(NvencError::ResourceNotRegistered)
        }
        sys::NVENCSTATUS_NV_ENC_ERR_RESOURCE_NOT_MAPPED => Err(NvencError::ResourceNotMapped),
        code => Err(NvencError::Unknown(code)),
    }
}
