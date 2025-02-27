package dev.vortex.api;

import com.google.errorprone.annotations.MustBeClosed;
import dev.vortex.jni.FFI;

public final class File extends BaseWrapped<FFI.FFIFile> {
    private File(FFI.FFIFile inner) {
        super(inner);
    }

    /**
     * Open a file at the provided path on the filesystem.
     */
    @MustBeClosed
    public static File open(String path) {
        return new File(FFI.FFIFile_open(path));
    }

    @Override
    public void close() {
        FFI.FFIFile_free(inner);
    }
}
