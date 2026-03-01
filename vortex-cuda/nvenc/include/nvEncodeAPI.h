/*
 * Minimal vendored NVENC API header for H.264 encoding.
 *
 * This header declares the subset of the NVIDIA Video Codec SDK 12.x API
 * needed for GPU-accelerated H.264 encoding with CUDA input buffers.
 *
 * The full header is available at:
 * https://developer.nvidia.com/video-codec-sdk
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef NV_ENCODE_API_H
#define NV_ENCODE_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- basic types ---------- */

typedef void *CUcontext;
typedef void *CUdeviceptr;

typedef struct GUID {
    uint32_t Data1;
    uint16_t Data2;
    uint16_t Data3;
    uint8_t  Data4[8];
} GUID;

/* ---------- version macros ---------- */

#define NVENCAPI_MAJOR_VERSION 12
#define NVENCAPI_MINOR_VERSION 2

/* ---------- status codes ---------- */

typedef enum NVENCSTATUS {
    NV_ENC_SUCCESS = 0,
    NV_ENC_ERR_NO_ENCODE_DEVICE = 1,
    NV_ENC_ERR_UNSUPPORTED_DEVICE = 2,
    NV_ENC_ERR_INVALID_ENCODERDEVICE = 3,
    NV_ENC_ERR_INVALID_DEVICE = 4,
    NV_ENC_ERR_DEVICE_NOT_EXIST = 5,
    NV_ENC_ERR_INVALID_PTR = 6,
    NV_ENC_ERR_INVALID_EVENT = 7,
    NV_ENC_ERR_INVALID_PARAM = 8,
    NV_ENC_ERR_INVALID_CALL = 9,
    NV_ENC_ERR_OUT_OF_MEMORY = 10,
    NV_ENC_ERR_ENCODER_NOT_INITIALIZED = 11,
    NV_ENC_ERR_UNSUPPORTED_PARAM = 12,
    NV_ENC_ERR_LOCK_BUSY = 13,
    NV_ENC_ERR_NOT_ENOUGH_BUFFER = 14,
    NV_ENC_ERR_INVALID_VERSION = 15,
    NV_ENC_ERR_MAP_FAILED = 16,
    NV_ENC_ERR_NEED_MORE_INPUT = 17,
    NV_ENC_ERR_ENCODER_BUSY = 18,
    NV_ENC_ERR_EVENT_NOT_REGISTERD = 19,
    NV_ENC_ERR_GENERIC = 20,
    NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY = 21,
    NV_ENC_ERR_UNIMPLEMENTED = 22,
    NV_ENC_ERR_RESOURCE_REGISTER_FAILED = 23,
    NV_ENC_ERR_RESOURCE_NOT_REGISTERED = 24,
    NV_ENC_ERR_RESOURCE_NOT_MAPPED = 25,
} NVENCSTATUS;

/* ---------- enums ---------- */

typedef enum NV_ENC_DEVICE_TYPE {
    NV_ENC_DEVICE_TYPE_DIRECTX = 0,
    NV_ENC_DEVICE_TYPE_CUDA = 1,
    NV_ENC_DEVICE_TYPE_OPENGL = 2,
} NV_ENC_DEVICE_TYPE;

typedef enum NV_ENC_INPUT_RESOURCE_TYPE {
    NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX = 0,
    NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 1,
    NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY = 2,
    NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX = 3,
} NV_ENC_INPUT_RESOURCE_TYPE;

typedef enum NV_ENC_BUFFER_FORMAT {
    NV_ENC_BUFFER_FORMAT_UNDEFINED = 0x00000000,
    NV_ENC_BUFFER_FORMAT_NV12     = 0x00000001,
    NV_ENC_BUFFER_FORMAT_YV12     = 0x00000010,
    NV_ENC_BUFFER_FORMAT_IYUV     = 0x00000100,
    NV_ENC_BUFFER_FORMAT_YUV444   = 0x00001000,
    NV_ENC_BUFFER_FORMAT_YUV420_10BIT = 0x00010000,
    NV_ENC_BUFFER_FORMAT_YUV444_10BIT = 0x00100000,
    NV_ENC_BUFFER_FORMAT_ARGB     = 0x01000000,
    NV_ENC_BUFFER_FORMAT_ARGB10   = 0x02000000,
    NV_ENC_BUFFER_FORMAT_AYUV     = 0x04000000,
    NV_ENC_BUFFER_FORMAT_ABGR     = 0x10000000,
    NV_ENC_BUFFER_FORMAT_ABGR10   = 0x20000000,
    NV_ENC_BUFFER_FORMAT_U8       = 0x40000000,
} NV_ENC_BUFFER_FORMAT;

typedef enum NV_ENC_PIC_TYPE {
    NV_ENC_PIC_TYPE_P               = 0,
    NV_ENC_PIC_TYPE_B               = 1,
    NV_ENC_PIC_TYPE_I               = 2,
    NV_ENC_PIC_TYPE_IDR             = 3,
    NV_ENC_PIC_TYPE_BI              = 4,
    NV_ENC_PIC_TYPE_SKIPPED         = 5,
    NV_ENC_PIC_TYPE_INTRA_REFRESH   = 6,
    NV_ENC_PIC_TYPE_NONREF_P        = 7,
    NV_ENC_PIC_TYPE_UNKNOWN         = 0xFF,
} NV_ENC_PIC_TYPE;

typedef enum NV_ENC_PIC_STRUCT {
    NV_ENC_PIC_STRUCT_FRAME = 0x01,
    NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM = 0x02,
    NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP = 0x03,
} NV_ENC_PIC_STRUCT;

typedef enum NV_ENC_TUNING_INFO {
    NV_ENC_TUNING_INFO_UNDEFINED         = 0,
    NV_ENC_TUNING_INFO_HIGH_QUALITY      = 1,
    NV_ENC_TUNING_INFO_LOW_LATENCY       = 2,
    NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY = 3,
    NV_ENC_TUNING_INFO_LOSSLESS          = 4,
} NV_ENC_TUNING_INFO;

typedef enum NV_ENC_PARAMS_RC_MODE {
    NV_ENC_PARAMS_RC_CONSTQP    = 0x0,
    NV_ENC_PARAMS_RC_VBR        = 0x1,
    NV_ENC_PARAMS_RC_CBR        = 0x2,
} NV_ENC_PARAMS_RC_MODE;

/* ---------- opaque handles ---------- */

typedef void *NV_ENC_INPUT_PTR;
typedef void *NV_ENC_OUTPUT_PTR;
typedef void *NV_ENC_REGISTERED_PTR;

/* ---------- structs ---------- */

typedef struct NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
    uint32_t          version;
    NV_ENC_DEVICE_TYPE deviceType;
    void              *device;
    uint32_t          reserved;
    uint32_t          apiVersion;
    uint32_t          reserved1[253];
    void              *reserved2[64];
} NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS;

typedef struct NV_ENC_PRESET_CONFIG {
    uint32_t version;
    uint32_t reserved[1024];
} NV_ENC_PRESET_CONFIG;

typedef struct NV_ENC_RC_PARAMS {
    NV_ENC_PARAMS_RC_MODE rateControlMode;
    uint32_t constQP_interP;
    uint32_t constQP_interB;
    uint32_t constQP_intra;
    uint32_t averageBitRate;
    uint32_t maxBitRate;
    uint32_t vbvBufferSize;
    uint32_t vbvInitialDelay;
    uint32_t reserved[8];
    uint32_t reserved2[248];
} NV_ENC_RC_PARAMS;

typedef struct NV_ENC_CONFIG_H264 {
    uint32_t enableStereoMVC;
    uint32_t hierarchicalPFrames;
    uint32_t hierarchicalBFrames;
    uint32_t outputBufferingPeriodSEI;
    uint32_t outputPictureTimingSEI;
    uint32_t outputAUD;
    uint32_t disableSPSPPS;
    uint32_t outputFramePackingSEI;
    uint32_t outputRecoveryPointSEI;
    uint32_t enableIntraRefresh;
    uint32_t enableConstrainedEncoding;
    uint32_t repeatSPSPPS;
    uint32_t enableVFR;
    uint32_t enableLTR;
    uint32_t qpPrimeYZeroTransformBypassFlag;
    uint32_t useConstrainedIntraPred;
    uint32_t enableFillerDataInsertion;
    uint32_t reserved[223];
    uint32_t reserved2[64];
} NV_ENC_CONFIG_H264;

typedef union NV_ENC_CODEC_CONFIG {
    NV_ENC_CONFIG_H264 h264Config;
    uint32_t reserved[512];
} NV_ENC_CODEC_CONFIG;

typedef struct NV_ENC_CONFIG {
    uint32_t           version;
    GUID               profileGUID;
    uint32_t           gopLength;
    int32_t            frameIntervalP;
    uint32_t           monoChromeEncoding;
    uint32_t           frameFieldMode;
    uint32_t           reserved_mvPrecision;
    NV_ENC_RC_PARAMS   rcParams;
    NV_ENC_CODEC_CONFIG encodeCodecConfig;
    uint32_t           reserved[278];
    void               *reserved2[64];
} NV_ENC_CONFIG;

typedef struct NV_ENC_INITIALIZE_PARAMS {
    uint32_t          version;
    GUID              encodeGUID;
    GUID              presetGUID;
    uint32_t          encodeWidth;
    uint32_t          encodeHeight;
    uint32_t          darWidth;
    uint32_t          darHeight;
    uint32_t          frameRateNum;
    uint32_t          frameRateDen;
    uint32_t          enableEncodeAsync;
    uint32_t          enablePTD;
    uint32_t          reportSliceOffsets;
    uint32_t          enableSubFrameWrite;
    uint32_t          enableExternalMEHints;
    uint32_t          enableMEOnlyMode;
    uint32_t          enableWeightedPrediction;
    uint32_t          enableOutputInVidmem;
    uint32_t          reservedBitFields;
    uint32_t          privDataSize;
    void              *privData;
    NV_ENC_CONFIG     *encodeConfig;
    uint32_t          maxEncodeWidth;
    uint32_t          maxEncodeHeight;
    uint32_t          reserved[281];
    void              *reserved2[64];
} NV_ENC_INITIALIZE_PARAMS;

typedef struct NV_ENC_REGISTER_RESOURCE {
    uint32_t                   version;
    NV_ENC_INPUT_RESOURCE_TYPE resourceType;
    uint32_t                   width;
    uint32_t                   height;
    uint32_t                   pitch;
    uint32_t                   subResourceIndex;
    void                       *resourceToRegister;
    NV_ENC_REGISTERED_PTR      registeredResource;
    NV_ENC_BUFFER_FORMAT       bufferFormat;
    uint32_t                   bufferUsage;
    uint32_t                   reserved[62];
    void                       *reserved2[64];
} NV_ENC_REGISTER_RESOURCE;

typedef struct NV_ENC_MAP_INPUT_RESOURCE {
    uint32_t              version;
    uint32_t              subResourceIndex;
    NV_ENC_REGISTERED_PTR registeredResource;
    NV_ENC_INPUT_PTR      mappedResource;
    NV_ENC_BUFFER_FORMAT  mappedBufferFmt;
    uint32_t              reserved1[62];
    void                  *reserved2[64];
} NV_ENC_MAP_INPUT_RESOURCE;

typedef struct NV_ENC_CREATE_BITSTREAM_BUFFER {
    uint32_t          version;
    uint32_t          size;
    uint32_t          memoryHeap;
    NV_ENC_OUTPUT_PTR bitstreamBuffer;
    void              *bitstreamBufferPtr;
    uint32_t          reserved[58];
    void              *reserved2[64];
} NV_ENC_CREATE_BITSTREAM_BUFFER;

typedef struct NV_ENC_PIC_PARAMS {
    uint32_t              version;
    uint32_t              inputWidth;
    uint32_t              inputHeight;
    uint32_t              inputPitch;
    uint32_t              encodePicFlags;
    uint32_t              frameIdx;
    uint64_t              inputTimeStamp;
    uint64_t              inputDuration;
    NV_ENC_INPUT_PTR      inputBuffer;
    NV_ENC_OUTPUT_PTR     outputBitstream;
    void                  *completionEvent;
    NV_ENC_BUFFER_FORMAT  bufferFmt;
    NV_ENC_PIC_STRUCT     pictureStruct;
    NV_ENC_PIC_TYPE       pictureType;
    NV_ENC_CODEC_CONFIG   codecPicParams;
    uint32_t              reserved[286];
    void                  *reserved2[64];
} NV_ENC_PIC_PARAMS;

#define NV_ENC_PIC_FLAG_EOS 0x1
#define NV_ENC_PIC_FLAG_FORCEIDR 0x4

typedef struct NV_ENC_LOCK_BITSTREAM {
    uint32_t          version;
    uint32_t          doNotWait;
    uint32_t          lkbdFlags;
    void              *reserved1;
    NV_ENC_OUTPUT_PTR outputBitstream;
    uint32_t          *sliceOffsets;
    uint32_t          frameIdx;
    uint32_t          hwEncodeStatus;
    uint32_t          numSlices;
    uint32_t          bitstreamSizeInBytes;
    uint64_t          outputTimeStamp;
    uint64_t          outputDuration;
    void              *bitstreamBufferPtr;
    NV_ENC_PIC_TYPE   pictureType;
    NV_ENC_PIC_STRUCT pictureStruct;
    uint32_t          frameAvgQP;
    uint32_t          frameSatd;
    uint32_t          ltrFrameIdx;
    uint32_t          ltrFrameBitmap;
    uint32_t          reserved[13];
    uint32_t          intraMBCount;
    uint32_t          interMBCount;
    int32_t           averageMVX;
    int32_t           averageMVY;
    uint32_t          reserved3[219];
    void              *reserved4[64];
} NV_ENC_LOCK_BITSTREAM;

/* ---------- function pointer table ---------- */

typedef struct NV_ENCODE_API_FUNCTION_LIST {
    uint32_t version;
    uint32_t reserved;

    NVENCSTATUS (*nvEncOpenEncodeSessionEx)
        (NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS *openSessionExParams, void **encoder);

    NVENCSTATUS (*nvEncGetEncodeGUIDCount)
        (void *encoder, uint32_t *encodeGUIDCount);

    NVENCSTATUS (*nvEncGetEncodeGUIDs)
        (void *encoder, GUID *GUIDs, uint32_t guidArraySize, uint32_t *GUIDCount);

    NVENCSTATUS (*nvEncGetEncodeProfileGUIDCount)
        (void *encoder, GUID encodeGUID, uint32_t *encodeProfileGUIDCount);

    NVENCSTATUS (*nvEncGetEncodeProfileGUIDs)
        (void *encoder, GUID encodeGUID, GUID *profileGUIDs,
         uint32_t guidArraySize, uint32_t *GUIDCount);

    NVENCSTATUS (*nvEncGetInputFormatCount)
        (void *encoder, GUID encodeGUID, uint32_t *inputFmtCount);

    NVENCSTATUS (*nvEncGetInputFormats)
        (void *encoder, GUID encodeGUID, NV_ENC_BUFFER_FORMAT *inputFmts,
         uint32_t inputFmtArraySize, uint32_t *inputFmtCount);

    NVENCSTATUS (*nvEncGetEncodeCaps)
        (void *encoder, GUID encodeGUID, void *capsParam, int32_t *capsVal);

    NVENCSTATUS (*nvEncGetEncodePresetCount)
        (void *encoder, GUID encodeGUID, uint32_t *encodePresetGUIDCount);

    NVENCSTATUS (*nvEncGetEncodePresetGUIDs)
        (void *encoder, GUID encodeGUID, GUID *presetGUIDs,
         uint32_t guidArraySize, uint32_t *GUIDCount);

    NVENCSTATUS (*nvEncGetEncodePresetConfigEx)
        (void *encoder, GUID encodeGUID, GUID presetGUID,
         NV_ENC_TUNING_INFO tuningInfo, NV_ENC_PRESET_CONFIG *presetConfig);

    NVENCSTATUS (*nvEncInitializeEncoder)
        (void *encoder, NV_ENC_INITIALIZE_PARAMS *createEncodeParams);

    NVENCSTATUS (*nvEncCreateInputBuffer)
        (void *encoder, void *createInputBufferParams);

    NVENCSTATUS (*nvEncDestroyInputBuffer)
        (void *encoder, NV_ENC_INPUT_PTR inputBuffer);

    NVENCSTATUS (*nvEncCreateBitstreamBuffer)
        (void *encoder, NV_ENC_CREATE_BITSTREAM_BUFFER *createBitstreamBufferParams);

    NVENCSTATUS (*nvEncDestroyBitstreamBuffer)
        (void *encoder, NV_ENC_OUTPUT_PTR bitstreamBuffer);

    NVENCSTATUS (*nvEncEncodePicture)
        (void *encoder, NV_ENC_PIC_PARAMS *encodePicParams);

    NVENCSTATUS (*nvEncLockBitstream)
        (void *encoder, NV_ENC_LOCK_BITSTREAM *lockBitstreamBufferParams);

    NVENCSTATUS (*nvEncUnlockBitstream)
        (void *encoder, NV_ENC_OUTPUT_PTR bitstreamBuffer);

    void *nvEncLockInputBuffer;
    void *nvEncUnlockInputBuffer;

    void *nvEncGetEncodeStats;
    void *nvEncGetSequenceParams;
    void *nvEncRegisterAsyncEvent;
    void *nvEncUnregisterAsyncEvent;

    NVENCSTATUS (*nvEncMapInputResource)
        (void *encoder, NV_ENC_MAP_INPUT_RESOURCE *mapInputResParams);

    NVENCSTATUS (*nvEncUnmapInputResource)
        (void *encoder, NV_ENC_INPUT_PTR mappedInputBuffer);

    NVENCSTATUS (*nvEncDestroyEncoder)
        (void *encoder);

    void *nvEncInvalidateRefFrames;
    void *nvEncOpenEncodeSession;

    NVENCSTATUS (*nvEncRegisterResource)
        (void *encoder, NV_ENC_REGISTER_RESOURCE *registerResParams);

    NVENCSTATUS (*nvEncUnregisterResource)
        (void *encoder, NV_ENC_REGISTERED_PTR registeredRes);

    void *nvEncReconfigureEncoder;
    void *reserved1;
    void *nvEncCreateMVBuffer;
    void *nvEncDestroyMVBuffer;
    void *nvEncRunMotionEstimationOnly;
    void *nvEncGetLastErrorString;
    void *nvEncSetIOCudaStreams;
    void *nvEncGetEncodePresetConfig;
    void *nvEncGetSequenceParamEx;
    void *nvEncLookaheadPicture;
    void *reserved2[277];
} NV_ENCODE_API_FUNCTION_LIST;

/* ---------- exported functions ---------- */

NVENCSTATUS NvEncodeAPICreateInstance(NV_ENCODE_API_FUNCTION_LIST *functionList);
NVENCSTATUS NvEncodeAPIGetMaxSupportedVersion(uint32_t *version);

#ifdef __cplusplus
}
#endif

#endif /* NV_ENCODE_API_H */
