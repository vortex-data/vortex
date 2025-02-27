package dev.vortex.api;

import dev.vortex.jni.FFI;

/**
 * Vortex logical type.
 */
public final class DType extends BaseWrapped<FFI.FFIDType> {
    private final Variant variant;

    public DType(FFI.FFIDType inner) {
        super(inner);
        this.variant = Variant.from(FFI.FFIDType_get(inner));
    }

    public Variant getVariant() {
        return variant;
    }

    public boolean isNullable() {
        return inner.getPointer().getByte(1) == 1;
    }

    @Override
    public String toString() {
        return "DType{" + this.variant.toString() + ", nullable=" + isNullable() + "}";
    }

    @Override
    public void close() {
        FFI.FFIDType_free(inner);
    }

    public enum Variant {
        NULL(0), BOOL(1), PRIMITIVE_U8(2), PRIMITIVE_U16(3), PRIMITIVE_U32(4), PRIMITIVE_U64(5), PRIMITIVE_I8(6), PRIMITIVE_I16(7), PRIMITIVE_I32(8), PRIMITIVE_I64(9), PRIMITIVE_F16(10), PRIMITIVE_F32(11), PRIMITIVE_F64(12), UTF8(13), BINARY(14), STRUCT(15), LIST(16), EXTENSION(17);;

        private int variant;

        Variant(int variant) {
            this.variant = variant;
        }

        public static Variant from(byte variant) {
            switch (variant) {
                case 0:
                    return NULL;
                case 1:
                    return BOOL;
                case 2:
                    return PRIMITIVE_U8;
                case 3:
                    return PRIMITIVE_U16;
                case 4:
                    return PRIMITIVE_U32;
                case 5:
                    return PRIMITIVE_U64;
                case 6:
                    return PRIMITIVE_I8;
                case 7:
                    return PRIMITIVE_I16;
                case 8:
                    return PRIMITIVE_I32;
                case 9:
                    return PRIMITIVE_I64;
                case 10:
                    return PRIMITIVE_F16;
                case 11:
                    return PRIMITIVE_F32;
                case 12:
                    return PRIMITIVE_F64;
                case 13:
                    return UTF8;
                case 14:
                    return BINARY;
                case 15:
                    return STRUCT;
                case 16:
                    return LIST;
                case 17:
                    return EXTENSION;
                default:
                    throw new IllegalArgumentException("Unknown DType variant: " + variant);
            }
        }
    }

}
