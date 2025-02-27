package dev.vortex.api;

import com.jakewharton.nopen.annotation.Open;
import dev.vortex.jni.FFI;

/**
 * Core Vortex array type that all logical arrays inherit from.
 */
@Open
public class BaseArray extends BaseWrapped<FFI.FFIArray> {
    private final DType dtype;

    public BaseArray(FFI.FFIArray inner) {
        super(inner);
        this.dtype = new DType(FFI.FFIArray_dtype(inner));
    }

    @Override
    public void close() {
        // Free all resources
        FFI.FFIArray_free(inner);

        // Free the DType.
        dtype.close();
    }

    /**
     * Get the length of the array.
     */
    public long getLength() {
        return FFI.FFIArray_len(inner);
    }

    public int getInt(int index) {
        return 0;
    }

    public long getLong(int index) {
        return 0L;
    }

    public boolean getBool(int index) {
        return false;
    }

    public float getFloat(int index) {
        return 0.0f;
    }

    public double getDouble(int index) {
        return 0.0;
    }
}
