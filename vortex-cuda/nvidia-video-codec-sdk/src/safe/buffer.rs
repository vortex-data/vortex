//! Defines traits and types for dealing with input and output buffers.

use std::{ffi::c_void, ptr};

use cudarc::driver::{DevicePtr, MappedBuffer};

use super::{api::ENCODE_API, encoder::Encoder, result::EncodeError, session::Session};
use crate::sys::nvEncodeAPI::{
    NV_ENC_BUFFER_FORMAT,
    NV_ENC_CREATE_BITSTREAM_BUFFER,
    NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
    NV_ENC_CREATE_INPUT_BUFFER,
    NV_ENC_CREATE_INPUT_BUFFER_VER,
    NV_ENC_INPUT_RESOURCE_TYPE,
    NV_ENC_LOCK_BITSTREAM,
    NV_ENC_LOCK_BITSTREAM_VER,
    NV_ENC_LOCK_INPUT_BUFFER,
    NV_ENC_LOCK_INPUT_BUFFER_VER,
    NV_ENC_MAP_INPUT_RESOURCE,
    NV_ENC_MAP_INPUT_RESOURCE_VER,
    NV_ENC_PIC_TYPE,
    NV_ENC_REGISTER_RESOURCE,
};

/// If a type implements this trait it means it is a valid input buffer
/// for the encoding API.
pub trait EncoderInput {
    /// Get the pitch (AKA stride) of the input resource.
    fn pitch(&self) -> u32;

    /// Get the handle of the input resource.
    fn handle(&mut self) -> *mut c_void;
}

/// If a type implements this trait it means it is a valid output buffer
/// for the encoding API.
pub trait EncoderOutput {
    /// Get the handle of the output resource.
    fn handle(&mut self) -> *mut c_void;
}

/// Functions for creating input and output buffers.
impl Session {
    /// Create a [`Buffer`].
    ///
    /// See [NVIDIA docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-video-encoder-api-prog-guide/index.html#creating-resources-required-to-hold-inputoutput-data).
    ///
    /// # Errors
    ///
    /// Could error if the `width`, `height`, or `buffer_format` is invalid,
    /// or if we run out of memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cudarc::driver::CudaContext;
    /// # use nvidia_video_codec_sdk::{
    /// #     sys::nvEncodeAPI::{
    /// #         NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
    /// #         NV_ENC_CODEC_H264_GUID,
    /// #         NV_ENC_PIC_PARAMS,
    /// #         NV_ENC_PIC_STRUCT,
    /// #     },
    /// #     Encoder, EncoderInitParams
    /// # };
    /// # const WIDTH: u32 = 1920;
    /// # const HEIGHT: u32 = 1080;
    /// //* Create encoder. *//
    /// # let cuda_ctx = CudaContext::new(0).unwrap();
    /// # let encoder = Encoder::initialize_with_cuda(cuda_ctx).unwrap();
    ///
    /// //* Set `encode_guid` and `buffer_format`, and check that H.264 encoding and the ARGB format are supported. *//
    /// # let encode_guid = NV_ENC_CODEC_H264_GUID;
    /// # let encode_guids = encoder.get_encode_guids().unwrap();
    /// # assert!(encode_guids.contains(&encode_guid));
    /// # let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
    /// # let input_formats = encoder.get_supported_input_formats(encode_guid).unwrap();
    /// # assert!(input_formats.contains(&buffer_format));
    ///
    /// //* Begin encoder session. *//
    /// # let mut initialize_params = EncoderInitParams::new(encode_guid, WIDTH, HEIGHT);
    /// # initialize_params.display_aspect_ratio(16, 9)
    /// #     .framerate(30, 1)
    /// #     .enable_picture_type_decision();
    /// # let session = encoder.start_session(
    /// #     buffer_format,
    /// #     initialize_params,
    /// # ).unwrap();
    ///
    /// // Create an input buffer.
    /// let _input_buffer = session
    ///     .create_input_buffer()
    ///     .unwrap();
    /// ```
    pub fn create_input_buffer(&self) -> Result<Buffer<'_>, EncodeError> {
        let mut create_input_buffer_params = NV_ENC_CREATE_INPUT_BUFFER {
            version: NV_ENC_CREATE_INPUT_BUFFER_VER,
            width: self.width,
            height: self.height,
            bufferFmt: self.buffer_format,
            inputBuffer: ptr::null_mut(),
            ..Default::default()
        };
        unsafe {
            (ENCODE_API.create_input_buffer)(self.encoder.ptr, &mut create_input_buffer_params)
        }
        .result(&self.encoder)?;
        Ok(Buffer {
            ptr: create_input_buffer_params.inputBuffer,
            pitch: self.width,
            encoder: &self.encoder,
        })
    }

    /// Create a [`Bitstream`].
    ///
    /// See [NVIDIA docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-video-encoder-api-prog-guide/index.html#creating-resources-required-to-hold-inputoutput-data).
    ///
    /// # Errors
    ///
    /// Could error is we run out of memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cudarc::driver::CudaContext;
    /// # use nvidia_video_codec_sdk::{
    /// #     sys::nvEncodeAPI::{
    /// #         NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
    /// #         NV_ENC_CODEC_H264_GUID,
    /// #         NV_ENC_PIC_PARAMS,
    /// #         NV_ENC_PIC_STRUCT,
    /// #     },
    /// #     Encoder, EncoderInitParams
    /// # };
    /// # const WIDTH: u32 = 1920;
    /// # const HEIGHT: u32 = 1080;
    /// //* Create encoder. *//
    /// # let cuda_ctx = CudaContext::new(0).unwrap();
    /// # let encoder = Encoder::initialize_with_cuda(cuda_ctx).unwrap();
    ///
    /// //* Set `encode_guid` and `buffer_format`, and check that H.264 encoding and the ARGB format are supported. *//
    /// # let encode_guid = NV_ENC_CODEC_H264_GUID;
    /// # let encode_guids = encoder.get_encode_guids().unwrap();
    /// # assert!(encode_guids.contains(&encode_guid));
    /// # let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
    /// # let input_formats = encoder.get_supported_input_formats(encode_guid).unwrap();
    /// # assert!(input_formats.contains(&buffer_format));
    ///
    /// //* Begin encoder session. *//
    /// # let mut initialize_params = EncoderInitParams::new(encode_guid, WIDTH, HEIGHT);
    /// # initialize_params.display_aspect_ratio(16, 9)
    /// #     .framerate(30, 1)
    /// #     .enable_picture_type_decision();
    /// # let session = encoder.start_session(
    /// #     buffer_format,
    /// #     initialize_params,
    /// # ).unwrap();
    ///
    /// // Create an output bitstream buffer.
    /// let _output_bitstream = session
    ///     .create_output_bitstream()
    ///     .unwrap();
    /// ```
    pub fn create_output_bitstream(&self) -> Result<Bitstream<'_>, EncodeError> {
        let mut create_bitstream_buffer_params = NV_ENC_CREATE_BITSTREAM_BUFFER {
            version: NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
            bitstreamBuffer: ptr::null_mut(),
            ..Default::default()
        };
        unsafe {
            (ENCODE_API.create_bitstream_buffer)(
                self.encoder.ptr,
                &mut create_bitstream_buffer_params,
            )
        }
        .result(&self.encoder)?;
        Ok(Bitstream {
            ptr: create_bitstream_buffer_params.bitstreamBuffer,
            encoder: &self.encoder,
        })
    }

    /// Create a [`RegisteredResource`] from a [`MappedBuffer`].
    ///
    /// See [`Session::register_generic_resource`].
    ///
    /// `pitch` should be set to the value obtained from `cuMemAllocPitch()`,
    /// or to the width in **bytes** (if this resource was created by using
    /// `cuMemAlloc()`). This value must be a multiple of 4.
    ///
    /// # Errors
    ///
    /// Could error if registration or mapping fails,
    /// if the resource is invalid, or if we run out of memory.
    pub fn register_cuda_resource(
        &self,
        pitch: u32,
        mapped_buffer: MappedBuffer,
    ) -> Result<RegisteredResource<'_, MappedBuffer>, EncodeError> {
        let stream = self.encoder.ctx.default_stream();
        let (device_ptr, _) = mapped_buffer.device_ptr(&stream);
        self.register_generic_resource(
            mapped_buffer,
            NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            device_ptr as *mut c_void,
            pitch,
        )
    }

    /// Create a [`RegisteredResource`].
    ///
    /// This function is generic in the marker. This is so that you can
    /// optionally put a value on the [`RegisteredResource`] to make sure that
    /// value does not get dropped while the resource is registered. You should
    /// prefer using specific functions for the resource you are registering,
    /// such as [`Session::register_cuda_resource`], when they are available.
    ///
    /// See [NVIDIA docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-video-encoder-api-prog-guide/index.html#input-buffers-allocated-externally).
    ///
    /// # Errors
    ///
    /// Could error if registration or mapping fails,
    /// if the resource is invalid, or if we run out of memory.
    pub fn register_generic_resource<T>(
        &self,
        marker: T,
        resource_type: NV_ENC_INPUT_RESOURCE_TYPE,
        resource_to_register: *mut c_void,
        pitch: u32,
    ) -> Result<RegisteredResource<'_, T>, EncodeError> {
        // Register resource.
        let mut register_resource_params = NV_ENC_REGISTER_RESOURCE::new(
            resource_type,
            self.width,
            self.height,
            resource_to_register,
            self.buffer_format,
        )
        .pitch(pitch);
        unsafe { (ENCODE_API.register_resource)(self.encoder.ptr, &mut register_resource_params) }
            .result(&self.encoder)?;
        let registered_resource = register_resource_params.registeredResource;

        // Map resource.
        let mut map_input_resource_params = NV_ENC_MAP_INPUT_RESOURCE {
            version: NV_ENC_MAP_INPUT_RESOURCE_VER,
            registeredResource: registered_resource,
            mappedResource: ptr::null_mut(),
            mappedBufferFmt: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED,
            ..Default::default()
        };
        unsafe {
            (ENCODE_API.map_input_resource)(self.encoder.ptr, &mut map_input_resource_params)
        }
        .result(&self.encoder)?;

        let mapped_resource = map_input_resource_params.mappedResource;
        Ok(RegisteredResource {
            reg_ptr: registered_resource,
            map_ptr: mapped_resource,
            pitch,
            encoder: &self.encoder,
            _marker: marker,
        })
    }
}

/// Abstraction around input buffer allocated using
/// the NVIDIA Video Encoder API.
///
/// The buffer is automatically destroyed when dropped.
#[derive(Debug)]
pub struct Buffer<'a> {
    pub(crate) ptr: *mut c_void,
    pitch: u32,
    encoder: &'a Encoder,
}

unsafe impl Send for Buffer<'_> {}

impl<'a> Buffer<'a> {
    /// Lock the input buffer.
    ///
    /// On a successful lock you get a [`BufferLock`] which can be used to write
    /// data to the input buffer. On drop, [`BufferLock`] will unlock the
    /// buffer.
    ///
    /// This function will block until a lock is acquired. For the non-blocking
    /// version see [`Buffer::try_lock`].
    ///
    /// See [NVIDIA docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-video-encoder-api-prog-guide/index.html#input-buffers-allocated-through-nvidia-video-encoder-interface).
    ///
    /// # Errors
    ///
    /// Could error if we run out of memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cudarc::driver::CudaContext;
    /// # use nvidia_video_codec_sdk::{
    /// #     sys::nvEncodeAPI::{
    /// #         NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
    /// #         NV_ENC_CODEC_H264_GUID,
    /// #         NV_ENC_PIC_PARAMS,
    /// #         NV_ENC_PIC_STRUCT,
    /// #     },
    /// #     Encoder, EncoderInitParams
    /// # };
    /// # const WIDTH: u32 = 1920;
    /// # const HEIGHT: u32 = 1080;
    /// # const DATA_LEN: usize = (WIDTH * HEIGHT * 4) as usize;
    /// //* Create encoder. *//
    /// # let cuda_ctx = CudaContext::new(0).unwrap();
    /// # let encoder = Encoder::initialize_with_cuda(cuda_ctx).unwrap();
    /// //* Set `encode_guid` and `buffer_format`, and check that H.264 encoding and the ARGB format are supported. *//
    /// # let encode_guid = NV_ENC_CODEC_H264_GUID;
    /// # let encode_guids = encoder.get_encode_guids().unwrap();
    /// # assert!(encode_guids.contains(&encode_guid));
    /// # let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
    /// # let input_formats = encoder.get_supported_input_formats(encode_guid).unwrap();
    /// # assert!(input_formats.contains(&buffer_format));
    /// //* Begin encoder session. *//
    /// # let mut initialize_params = EncoderInitParams::new(encode_guid, WIDTH, HEIGHT);
    /// # initialize_params.display_aspect_ratio(16, 9)
    /// #     .framerate(30, 1)
    /// #     .enable_picture_type_decision();
    /// # let session = encoder.start_session(
    /// #     buffer_format,
    /// #     initialize_params,
    /// # ).unwrap();
    ///
    /// // Create an input buffer.
    /// let mut input_buffer = session
    ///     .create_input_buffer()
    ///     .unwrap();
    /// unsafe { input_buffer.lock().unwrap().write(&[0; DATA_LEN]) };
    /// ```
    pub fn lock<'b>(&'b mut self) -> Result<BufferLock<'b, 'a>, EncodeError> {
        self.lock_inner(true)
    }

    /// Non-blocking version of [`Buffer::lock`]. See it for more info.
    ///
    /// This function will return an error with
    /// [`ErrorKind::EncoderBusy`](super::ErrorKind::EncoderBusy) or
    /// [`ErrorKind::LockBusy`](super::ErrorKind::LockBusy) if the lock is being
    /// used. The NVIDIA documentation from the header file is unclear about
    /// this.
    ///
    /// # Errors
    ///
    /// Could error if we run out of memory.
    ///
    /// If this returns an error with
    /// [`ErrorKind::EncoderBusy`](super::ErrorKind::EncoderBusy) or
    /// [`ErrorKind::LockBusy`](super::ErrorKind::LockBusy) then that means the
    /// lock is still busy and the client should retry in a few
    /// milliseconds.
    pub fn try_lock<'b>(&'b mut self) -> Result<BufferLock<'b, 'a>, EncodeError> {
        self.lock_inner(false)
    }

    #[inline]
    fn lock_inner<'b>(&'b mut self, wait: bool) -> Result<BufferLock<'b, 'a>, EncodeError> {
        let mut lock_input_buffer_params = NV_ENC_LOCK_INPUT_BUFFER {
            version: NV_ENC_LOCK_INPUT_BUFFER_VER,
            inputBuffer: self.ptr,
            ..Default::default()
        };
        if !wait {
            lock_input_buffer_params.set_doNotWait(1);
        }
        unsafe { (ENCODE_API.lock_input_buffer)(self.encoder.ptr, &mut lock_input_buffer_params) }
            .result(self.encoder)?;

        let data_ptr = lock_input_buffer_params.bufferDataPtr;
        let pitch = lock_input_buffer_params.pitch;
        self.pitch = pitch;

        Ok(BufferLock {
            buffer: self,
            data_ptr,
            pitch,
        })
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe { (ENCODE_API.destroy_input_buffer)(self.encoder.ptr, self.ptr) }
            .result(self.encoder)
            .expect("The encoder and buffer pointers should be valid.");
    }
}

impl EncoderInput for Buffer<'_> {
    fn pitch(&self) -> u32 {
        self.pitch
    }

    fn handle(&mut self) -> *mut c_void {
        self.ptr
    }
}

/// An RAII lock on the input buffer.
///
/// This type is created via [`Buffer::lock`] or [`Buffer::try_lock`].
/// The purpose of this type is similar to [`std::sync::MutexGuard`] -
/// it automatically unlocks the buffer when the lock goes out of scope.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct BufferLock<'a, 'b> {
    buffer: &'a Buffer<'b>,
    data_ptr: *mut c_void,
    #[allow(dead_code)]
    pitch: u32,
}

impl BufferLock<'_, '_> {
    /// Write data to the buffer.
    ///
    /// # Safety
    ///
    /// The size of the data should be less or equal to the size of the buffer.
    /// The size of the buffer depends on the width, height, and buffer format.
    ///
    /// The user should also account for pitch, the data is written
    /// contiguously.
    pub unsafe fn write(&mut self, data: &[u8]) {
        // TODO: Make this safe by doing checks.
        // - Check that length of data fits (depends on format).
        // - Write pitched?
        data.as_ptr()
            .copy_to(self.data_ptr.cast::<u8>(), data.len());
    }
}

impl Drop for BufferLock<'_, '_> {
    fn drop(&mut self) {
        unsafe { (ENCODE_API.unlock_input_buffer)(self.buffer.encoder.ptr, self.buffer.ptr) }
            .result(self.buffer.encoder)
            .expect("The encoder and buffer pointers should be valid.");
    }
}

/// Abstraction around the output bitstream buffer that
/// is used as the output of the encoding.
///
/// The buffer is automatically destroyed when dropped.
#[derive(Debug)]
pub struct Bitstream<'a> {
    pub(crate) ptr: *mut c_void,
    encoder: &'a Encoder,
}

unsafe impl Send for Bitstream<'_> {}

impl Bitstream<'_> {
    /// Lock the output bitstream.
    ///
    /// On a successful lock you get a [`BitstreamLock`] which can be used to
    /// access the bitstream data as well as any other information the
    /// encoder provides when locking a bitstream.
    ///
    /// This function will block until a lock is acquired. For the non-blocking
    /// version see [`Bitstream::try_lock`].
    ///
    /// See [NVIDIA docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-video-encoder-api-prog-guide/index.html#retrieving-encoded-output).
    ///
    /// # Errors
    ///
    /// Could error if we run out of memory.
    pub fn lock(&mut self) -> Result<BitstreamLock<'_, '_>, EncodeError> {
        self.lock_inner(true)
    }

    /// Non-blocking version of [`Bitstream::lock`]. See it for more info.
    ///
    /// This function will return an error with
    /// [`ErrorKind::LockBusy`](super::ErrorKind::LockBusy) if the
    /// lock is currently busy.
    ///
    /// # Errors
    ///
    /// Could error if we run out of memory.
    ///
    /// An error with [`ErrorKind::LockBusy`](super::ErrorKind::LockBusy) could
    /// be returned if the lock is currently busy. This is a recoverable
    /// error and the client should retry in a few milliseconds.
    pub fn try_lock(&mut self) -> Result<BitstreamLock<'_, '_>, EncodeError> {
        self.lock_inner(false)
    }

    fn lock_inner(&mut self, wait: bool) -> Result<BitstreamLock<'_, '_>, EncodeError> {
        // Lock bitstream.
        let mut lock_bitstream_buffer_params = NV_ENC_LOCK_BITSTREAM {
            version: NV_ENC_LOCK_BITSTREAM_VER,
            outputBitstream: self.ptr,
            ..Default::default()
        };
        if !wait {
            lock_bitstream_buffer_params.set_doNotWait(1);
        }
        unsafe { (ENCODE_API.lock_bitstream)(self.encoder.ptr, &mut lock_bitstream_buffer_params) }
            .result(self.encoder)?;

        // Get data.
        let data_ptr = lock_bitstream_buffer_params.bitstreamBufferPtr;
        let data_size = lock_bitstream_buffer_params.bitstreamSizeInBytes as usize;
        let data = unsafe { std::slice::from_raw_parts_mut(data_ptr.cast::<u8>(), data_size) };

        Ok(BitstreamLock {
            bitstream: self,
            data,
            frame_index: lock_bitstream_buffer_params.frameIdx,
            timestamp: lock_bitstream_buffer_params.outputTimeStamp,
            duration: lock_bitstream_buffer_params.outputDuration,
            picture_type: lock_bitstream_buffer_params.pictureType,
        })
    }
}

impl Drop for Bitstream<'_> {
    fn drop(&mut self) {
        unsafe { (ENCODE_API.destroy_bitstream_buffer)(self.encoder.ptr, self.ptr) }
            .result(self.encoder)
            .expect("The encoder and bitstream pointers should be valid.");
    }
}

impl EncoderOutput for Bitstream<'_> {
    fn handle(&mut self) -> *mut c_void {
        self.ptr
    }
}

/// An RAII lock on the output bitstream buffer.
///
/// This type is created via [`Bitstream::lock`] or [`Bitstream::try_lock`].
/// The purpose of this type is similar to [`std::sync::MutexGuard`] -
/// it automatically unlocks the buffer when the lock goes out of scope.
#[derive(Debug)]
pub struct BitstreamLock<'a, 'b> {
    bitstream: &'a Bitstream<'b>,
    data: &'a [u8],
    // statistics and other info
    frame_index: u32,
    timestamp: u64,
    duration: u64,
    picture_type: NV_ENC_PIC_TYPE,
    // TODO: other fields
}

impl BitstreamLock<'_, '_> {
    /// Getter for the data contained in the output bitstream.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Getter for the frame index.
    #[must_use]
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Getter for the timestamp.
    #[must_use]
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Getter for the duration.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.duration
    }

    /// Getter for the picture type.
    #[must_use]
    pub fn picture_type(&self) -> NV_ENC_PIC_TYPE {
        self.picture_type
    }
}

impl Drop for BitstreamLock<'_, '_> {
    fn drop(&mut self) {
        unsafe { (ENCODE_API.unlock_bitstream)(self.bitstream.encoder.ptr, self.bitstream.ptr) }
            .result(self.bitstream.encoder)
            .expect("The encoder and bitstream pointers should be valid.");
    }
}

/// Abstraction for a registered and mapped external resource.
///
/// The Encoder API exposes a way to use input buffers allocated externally,
/// for example through CUDA or OpenGL.
///
/// The buffer is automatically unmapped and unregistered when dropped.
/// The external buffer memory should still be properly destroyed by the client.
#[derive(Debug)]
pub struct RegisteredResource<'a, T> {
    pub(crate) reg_ptr: *mut c_void,
    pub(crate) map_ptr: *mut c_void,
    pitch: u32,
    encoder: &'a Encoder,
    // A generic marker to make sure the external resources are dropped
    // after the resource is unregistered.
    _marker: T,
}

unsafe impl Send for RegisteredResource<'_, MappedBuffer> {}

/// Automatically unmap and unregister the external resource
/// when it goes out of scope.
impl<T> Drop for RegisteredResource<'_, T> {
    fn drop(&mut self) {
        // Unmapping resource.
        unsafe { (ENCODE_API.unmap_input_resource)(self.encoder.ptr, self.map_ptr) }
            .result(self.encoder)
            .expect("The encoder pointer and map handle should be valid.");
        // Unregister resource.
        unsafe { (ENCODE_API.unregister_resource)(self.encoder.ptr, self.reg_ptr) }
            .result(self.encoder)
            .expect("The encoder pointer and resource handle should be valid.");
    }
}

impl<T> EncoderInput for RegisteredResource<'_, T> {
    fn pitch(&self) -> u32 {
        self.pitch
    }

    fn handle(&mut self) -> *mut c_void {
        self.map_ptr
    }
}
