from math import iota
from sys import exit, has_accelerator

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx

alias num_elements_for = 1048576

fn kernel_frame_of_reference_i32(
    values_in_out: UnsafePointer[Int32, MutAnyOrigin],
    size: Int,
    reference: Int32,
):
    # Calculate the global thread index within the entire grid
    # Each thread processes one element of the array
    var idx = block_idx.x * block_dim.x + thread_idx.x

    # Bounds checking: ensure we don't access memory beyond the array size
    # This is crucial when the number of threads doesn't exactly match array size
    if idx < UInt(size):
        # Each thread adds the reference to its corresponding array element
        # This operation happens in parallel across all GPU threads
        values_in_out[idx] = values_in_out[idx] + reference


@export("mojo_frame_of_reference", ABI="C")
def frame_of_reference():
    @parameter
    if not has_accelerator():
        return
    else:
        with DeviceContext() as ctx:
            # Create a buffer in host CPU memory
            host_buffer = ctx.enqueue_create_host_buffer[DType.int32](
                num_elements_for
            )

            ctx.synchronize()

            # Fill the host buffer with sequential numbers
            iota(host_buffer.unsafe_ptr(), num_elements_for)

            # Create a buffer in device GPU memory
            device_buffer = ctx.enqueue_create_buffer[DType.int32](num_elements_for)

            # Copy data from host memory to device memory for GPU processing.
            ctx.enqueue_copy(src_buf=host_buffer, dst_buf=device_buffer)

            # Compile the kernel_frame_of_reference_i32 kernel
            for_kernel = ctx.compile_function_checked[
                kernel_frame_of_reference_i32, kernel_frame_of_reference_i32
            ]()

            # GPU kernel with dimensions for 1M elements.
            var threads_per_block = 256
            var num_blocks = (num_elements_for + threads_per_block - 1) // threads_per_block

            ctx.enqueue_function_checked(
                for_kernel,
                device_buffer,
                num_elements_for,
                Int32(42),  # reference value
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

            # Copy results back from device to host memory
            ctx.enqueue_copy(src_buf=device_buffer, dst_buf=host_buffer)

            ctx.synchronize()
