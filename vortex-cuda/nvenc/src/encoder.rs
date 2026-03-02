// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Safe wrapper around the NVENC encoder API.
//!
//! Provides a high-level interface for encoding NV12 frames from CUDA device
//! memory to H.264 bitstream.

use std::ptr;

use crate::error::NvencError;
use crate::error::check_status;
use crate::nvenc_library;
use crate::sys;

/// A registered CUDA resource that can be used as encoder input.
pub struct RegisteredResource {
    encoder: *mut std::ffi::c_void,
    registered_ptr: sys::NV_ENC_REGISTERED_PTR,
    fn_table: *const sys::NV_ENCODE_API_FUNCTION_LIST,
}

impl Drop for RegisteredResource {
    fn drop(&mut self) {
        // SAFETY: We have exclusive ownership and the encoder handle is valid.
        unsafe {
            if let Some(unregister) = (*self.fn_table).nvEncUnregisterResource {
                let _ = unregister(self.encoder, self.registered_ptr);
            }
        }
    }
}

/// NVENC H.264 encoder wrapping a CUDA device context.
///
/// Encodes NV12 frames from GPU memory to H.264 NAL units.
pub struct NvEncoder {
    encoder: *mut std::ffi::c_void,
    fn_table: Box<sys::NV_ENCODE_API_FUNCTION_LIST>,
    bitstream_buffer: sys::NV_ENC_OUTPUT_PTR,
    width: u32,
    height: u32,
}

// SAFETY: The NVENC encoder handle and function table are thread-safe.
// NVENC serializes operations internally.
unsafe impl Send for NvEncoder {}

impl NvEncoder {
    /// Creates a new NVENC H.264 encoder.
    ///
    /// # Arguments
    ///
    /// * `cu_context` - CUDA context pointer (from cudarc)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `fps` - Target frame rate
    /// * `bitrate` - Target bitrate in bits per second
    pub fn new(
        cu_context: sys::CUcontext,
        width: u32,
        height: u32,
        fps: u32,
        bitrate: u32,
    ) -> Result<Self, NvencError> {
        let library = nvenc_library()?;

        // Check max supported API version.
        // NvEncodeAPIGetMaxSupportedVersion returns version as (major << 4) | minor,
        // which differs from the NVENCAPI_VERSION macro format.
        let mut max_version: u32 = 0;
        let status = unsafe { library.NvEncodeAPIGetMaxSupportedVersion(&raw mut max_version) };
        check_status(status).map_err(|e| {
            NvencError::DetailedError(format!("NvEncodeAPIGetMaxSupportedVersion failed: {e}"))
        })?;
        let max_major = max_version >> 4;
        let max_minor = max_version & 0xF;
        let our_major = sys::NVENCAPI_MAJOR_VERSION;
        let our_minor = sys::NVENCAPI_MINOR_VERSION;
        let our_packed = (our_major << 4) | our_minor;
        if max_version < our_packed {
            return Err(NvencError::DetailedError(format!(
                "Driver NVENC API {max_major}.{max_minor} is older than required {our_major}.{our_minor}"
            )));
        }

        // Initialize function table
        let mut fn_table: Box<sys::NV_ENCODE_API_FUNCTION_LIST> =
            Box::new(unsafe { std::mem::zeroed() });
        fn_table.version = crate::NV_ENCODE_API_FUNCTION_LIST_VER;

        // SAFETY: fn_table is valid and zeroed with correct version set.
        let status = unsafe { library.NvEncodeAPICreateInstance(&raw mut *fn_table) };
        check_status(status).map_err(|e| {
            NvencError::DetailedError(format!("NvEncodeAPICreateInstance failed: {e}"))
        })?;

        // Open encode session
        let mut session_params: sys::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS =
            unsafe { std::mem::zeroed() };
        session_params.version = crate::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        session_params.deviceType = sys::NV_ENC_DEVICE_TYPE_CUDA;
        session_params.device = cu_context;
        session_params.apiVersion = crate::NVENCAPI_VERSION;

        let mut encoder: *mut std::ffi::c_void = ptr::null_mut();
        let open_fn = fn_table
            .nvEncOpenEncodeSessionEx
            .ok_or(NvencError::Generic)?;
        // SAFETY: session_params is properly initialized.
        let status = unsafe { open_fn(&raw mut session_params, &raw mut encoder) };
        check_status(status).map_err(|e| {
            NvencError::DetailedError(format!("nvEncOpenEncodeSessionEx failed: {e}"))
        })?;

        // Configure encode params
        let mut encode_config: sys::NV_ENC_CONFIG = unsafe { std::mem::zeroed() };
        encode_config.version = crate::NV_ENC_CONFIG_VER;
        encode_config.profileGUID = crate::NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
        encode_config.gopLength = fps; // 1-second GOPs
        encode_config.frameIntervalP = 1; // No B-frames
        encode_config.rcParams.rateControlMode = sys::NV_ENC_PARAMS_RC_CBR;
        encode_config.rcParams.averageBitRate = bitrate;
        encode_config.rcParams.maxBitRate = bitrate;
        encode_config.rcParams.vbvBufferSize = bitrate / fps; // 1 frame
        encode_config.rcParams.vbvInitialDelay = bitrate / fps;
        // Enable repeat SPS/PPS for stream joining
        // SAFETY: We know we're configuring H.264, so h264Config is the active union variant.
        unsafe {
            encode_config
                .encodeCodecConfig
                .h264Config
                .set_repeatSPSPPS(1);
        }

        let mut init_params: sys::NV_ENC_INITIALIZE_PARAMS = unsafe { std::mem::zeroed() };
        init_params.version = crate::NV_ENC_INITIALIZE_PARAMS_VER;
        init_params.encodeGUID = crate::NV_ENC_CODEC_H264_GUID;
        init_params.presetGUID = crate::NV_ENC_PRESET_P4_GUID;
        init_params.encodeWidth = width;
        init_params.encodeHeight = height;
        init_params.darWidth = width;
        init_params.darHeight = height;
        init_params.frameRateNum = fps;
        init_params.frameRateDen = 1;
        init_params.enablePTD = 1; // Picture type decision by encoder
        init_params.encodeConfig = &raw mut encode_config;

        let init_fn = fn_table.nvEncInitializeEncoder.ok_or(NvencError::Generic)?;
        // SAFETY: init_params is properly initialized with valid config.
        let status = unsafe { init_fn(encoder, &raw mut init_params) };
        check_status(status)?;

        // Create output bitstream buffer
        let mut bs_params: sys::NV_ENC_CREATE_BITSTREAM_BUFFER = unsafe { std::mem::zeroed() };
        bs_params.version = crate::NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

        let create_bs_fn = fn_table
            .nvEncCreateBitstreamBuffer
            .ok_or(NvencError::Generic)?;
        // SAFETY: bs_params is properly initialized.
        let status = unsafe { create_bs_fn(encoder, &raw mut bs_params) };
        check_status(status)?;

        Ok(Self {
            encoder,
            fn_table,
            bitstream_buffer: bs_params.bitstreamBuffer,
            width,
            height,
        })
    }

    /// Registers a CUDA device pointer as an encoder input resource.
    ///
    /// The pointer must remain valid for the lifetime of the returned resource.
    ///
    /// # Arguments
    ///
    /// * `device_ptr` - CUDA device pointer to NV12 frame data
    /// * `pitch` - Row pitch in bytes (typically width for NV12)
    pub fn register_input(
        &mut self,
        device_ptr: u64,
        pitch: u32,
    ) -> Result<RegisteredResource, NvencError> {
        let mut reg: sys::NV_ENC_REGISTER_RESOURCE = unsafe { std::mem::zeroed() };
        reg.version = crate::NV_ENC_REGISTER_RESOURCE_VER;
        reg.resourceType = sys::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        reg.width = self.width;
        reg.height = self.height;
        reg.pitch = pitch;
        reg.resourceToRegister = device_ptr as *mut std::ffi::c_void;
        reg.bufferFormat = sys::NV_ENC_BUFFER_FORMAT_NV12;

        let register_fn = self
            .fn_table
            .nvEncRegisterResource
            .ok_or(NvencError::Generic)?;
        // SAFETY: reg is properly initialized with valid device pointer.
        let status = unsafe { register_fn(self.encoder, &raw mut reg) };
        check_status(status)?;

        Ok(RegisteredResource {
            encoder: self.encoder,
            registered_ptr: reg.registeredResource,
            fn_table: &raw const *self.fn_table,
        })
    }

    /// Encodes a single frame from the registered resource.
    ///
    /// Returns the encoded H.264 NAL units as a byte vector.
    pub fn encode_frame(&mut self, resource: &RegisteredResource) -> Result<Vec<u8>, NvencError> {
        // Map input resource
        let mut map_params: sys::NV_ENC_MAP_INPUT_RESOURCE = unsafe { std::mem::zeroed() };
        map_params.version = crate::NV_ENC_MAP_INPUT_RESOURCE_VER;
        map_params.registeredResource = resource.registered_ptr;

        let map_fn = self
            .fn_table
            .nvEncMapInputResource
            .ok_or(NvencError::Generic)?;
        // SAFETY: map_params references a valid registered resource.
        let status = unsafe { map_fn(self.encoder, &raw mut map_params) };
        check_status(status)?;

        // Encode
        let mut pic_params: sys::NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        pic_params.version = crate::NV_ENC_PIC_PARAMS_VER;
        pic_params.inputWidth = self.width;
        pic_params.inputHeight = self.height;
        pic_params.inputPitch = self.width;
        pic_params.inputBuffer = map_params.mappedResource;
        pic_params.outputBitstream = self.bitstream_buffer;
        pic_params.bufferFmt = map_params.mappedBufferFmt;
        pic_params.pictureStruct = sys::NV_ENC_PIC_STRUCT_FRAME;

        let encode_fn = self
            .fn_table
            .nvEncEncodePicture
            .ok_or(NvencError::Generic)?;
        // SAFETY: pic_params references valid mapped resource and bitstream buffer.
        let status = unsafe { encode_fn(self.encoder, &raw mut pic_params) };
        // Unmap first, then check encode status
        let unmap_fn = self
            .fn_table
            .nvEncUnmapInputResource
            .ok_or(NvencError::Generic)?;
        // SAFETY: unmapping the previously mapped resource.
        let unmap_status = unsafe { unmap_fn(self.encoder, map_params.mappedResource) };

        check_status(status)?;
        check_status(unmap_status)?;

        // Lock and copy bitstream
        self.lock_and_copy_bitstream()
    }

    /// Flushes the encoder, retrieving any remaining frames.
    ///
    /// Should be called after all frames have been submitted.
    pub fn flush(&mut self) -> Result<Option<Vec<u8>>, NvencError> {
        let mut pic_params: sys::NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        pic_params.version = crate::NV_ENC_PIC_PARAMS_VER;
        pic_params.encodePicFlags = sys::NV_ENC_PIC_FLAG_EOS;

        let encode_fn = self
            .fn_table
            .nvEncEncodePicture
            .ok_or(NvencError::Generic)?;
        // SAFETY: EOS flush with zeroed params.
        let status = unsafe { encode_fn(self.encoder, &raw mut pic_params) };

        match status {
            sys::NV_ENC_SUCCESS => {
                let data = self.lock_and_copy_bitstream()?;
                Ok(Some(data))
            }
            sys::NV_ENC_ERR_NEED_MORE_INPUT => Ok(None),
            _ => {
                check_status(status)?;
                Ok(None)
            }
        }
    }

    fn lock_and_copy_bitstream(&mut self) -> Result<Vec<u8>, NvencError> {
        let mut lock_params: sys::NV_ENC_LOCK_BITSTREAM = unsafe { std::mem::zeroed() };
        lock_params.version = crate::NV_ENC_LOCK_BITSTREAM_VER;
        lock_params.outputBitstream = self.bitstream_buffer;

        let lock_fn = self
            .fn_table
            .nvEncLockBitstream
            .ok_or(NvencError::Generic)?;
        // SAFETY: lock_params references valid bitstream buffer.
        let status = unsafe { lock_fn(self.encoder, &raw mut lock_params) };
        check_status(status)?;

        let size = lock_params.bitstreamSizeInBytes as usize;
        let mut data = vec![0u8; size];

        // SAFETY: bitstreamBufferPtr is valid for bitstreamSizeInBytes after lock.
        unsafe {
            ptr::copy_nonoverlapping(
                lock_params.bitstreamBufferPtr as *const u8,
                data.as_mut_ptr(),
                size,
            );
        }

        let unlock_fn = self
            .fn_table
            .nvEncUnlockBitstream
            .ok_or(NvencError::Generic)?;
        // SAFETY: unlocking previously locked bitstream.
        let status = unsafe { unlock_fn(self.encoder, self.bitstream_buffer) };
        check_status(status)?;

        Ok(data)
    }
}

impl Drop for NvEncoder {
    fn drop(&mut self) {
        // SAFETY: Destroying the encoder session we created.
        unsafe {
            if let Some(destroy_bs) = self.fn_table.nvEncDestroyBitstreamBuffer {
                let _ = destroy_bs(self.encoder, self.bitstream_buffer);
            }
            if let Some(destroy) = self.fn_table.nvEncDestroyEncoder {
                let _ = destroy(self.encoder);
            }
        }
    }
}
