use std::{
    collections::VecDeque,
    fs::OpenOptions,
    io::Write,
    path::Path,
    sync::Arc,
    thread,
    time::Duration,
};

use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_H264_GUID},
    EncodeError,
    Encoder,
    EncoderInitParams,
    ErrorKind,
};

fn encode_blanks<P: AsRef<Path>>(
    cuda_ctx: Arc<CudaContext>,
    file_path: Option<P>,
) -> Result<(), EncodeError> {
    const FRAMES: usize = 128;
    const BUFFERS: usize = 16;
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;
    const FRAMERATE: u32 = 30;
    const BUFFER_FORMAT: NV_ENC_BUFFER_FORMAT = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB;
    const ENCODE_GUID: GUID = NV_ENC_CODEC_H264_GUID;
    // The size should be adjusted depending on the buffer format and pitch/stride.
    #[allow(clippy::large_stack_arrays)]
    const FRAME: [u8; (WIDTH * HEIGHT * 4) as usize] = [255; (WIDTH * HEIGHT * 4) as usize];

    let mut output = file_path.map(|path| {
        OpenOptions::new()
            .create(true)
            .write(true)
            .open(path)
            .expect("Path should be valid.")
    });

    // Initialize encoder.
    let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;
    let mut initialize_params = EncoderInitParams::new(ENCODE_GUID, WIDTH, HEIGHT);
    initialize_params
        .enable_picture_type_decision()
        .framerate(FRAMERATE, 1);
    let session = encoder.start_session(BUFFER_FORMAT, initialize_params)?;

    // Create input and output buffers.
    let mut input_buffers = (0..BUFFERS)
        .map(|_| session.create_input_buffer())
        .collect::<Result<Vec<_>, _>>()?;
    let mut output_bitstreams = (0..BUFFERS)
        .map(|_| session.create_output_bitstream())
        .collect::<Result<Vec<_>, _>>()?;
    // We will use this queue to mark which buffers are in-use.
    let mut in_use = VecDeque::with_capacity(BUFFERS);

    // Encode frames.
    'next_frame: for _ in 0..FRAMES {
        assert_eq!(input_buffers.len() + in_use.len(), BUFFERS);
        assert_eq!(output_bitstreams.len() + in_use.len(), BUFFERS);

        // Get an input and output buffer.
        let mut input_buffer = input_buffers
            .pop()
            .expect("There should be enough buffers.");
        let mut output_bitstream = output_bitstreams
            .pop()
            .expect("There should be enough buffers.");

        // Write a frame to the input buffer.
        unsafe { input_buffer.lock()?.write(&FRAME) };

        // Encode the frame.
        'encode: loop {
            match session.encode_picture(
                &mut input_buffer,
                &mut output_bitstream,
                Default::default(),
            ) {
                Ok(()) => {
                    // Success! Mark that these buffers are in-use.
                    in_use.push_back((input_buffer, output_bitstream));
                    break 'encode;
                }
                Err(e) if e.kind() == ErrorKind::EncoderBusy => {
                    // Encoder is busy, so let's just wait for a bit.
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) if e.kind() == ErrorKind::NeedMoreInput => {
                    // Encoder needs more input; mark that these buffers are in-use
                    // and skip to the next frame.
                    in_use.push_back((input_buffer, output_bitstream));
                    continue 'next_frame;
                }
                Err(e) => return Err(e),
            }
        }

        // In an attempt to speed things up, don't try to lock output bitstreams
        // immediately, but instead delay as long as possible.
        if in_use.len() < BUFFERS {
            continue;
        }

        // Get data out of bitstream, and put buffers back.
        let (in_buf, mut out_buf) = in_use
            .pop_front()
            .expect("There should be at least one element since that was just checked.");
        let lock = out_buf.lock()?;
        if let Some(file) = output.as_mut() {
            file.write_all(lock.data()).unwrap();
        }
        drop(lock);
        input_buffers.push(in_buf);
        output_bitstreams.push(out_buf);
    }

    // Finish reading the rest of the bitstream buffers.
    for (_, mut out_buf) in in_use {
        let lock = out_buf.lock()?;
        if let Some(file) = output.as_mut() {
            file.write_all(lock.data()).unwrap();
        }
    }

    Ok(())
}

#[test]
fn encoder_works() {
    encode_blanks::<&str>(
        CudaContext::new(0).expect("CUDA should be installed."),
        None,
    )
    .unwrap();
}

#[test]
fn encode_in_parallel() {
    std::thread::scope(|scope| {
        let cuda_ctx = CudaContext::new(0).expect("CUDA should be installed.");
        for _ in 0..4 {
            let thread_cuda_ctx = cuda_ctx.clone();
            scope.spawn(|| encode_blanks::<&str>(thread_cuda_ctx, None).unwrap());
        }
    });
}
