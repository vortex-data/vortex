// Standalone nvCOMP hardware-decompression-engine baseline (Deflate + LZ4).
// Compresses a raw byte file with nvCOMP, then times decompression on the
// dedicated hardware Decompression Engine (backend=HARDWARE). Reports decode
// GiB/s over the *uncompressed* bytes (directly comparable to OnPair decode).
//
// GOOD-BASELINE DEFAULTS (do not weaken):
//   - Deflate compress algorithm = 5 (max ratio). The SDK default (algo=1) is
//     "low compression ratio" and understates Deflate badly — see below.
//   - chunk = 256 KiB: near-optimal DE throughput. Ratio is NOT chunk-sensitive
//     (Deflate caps its back-reference window at 32 KiB), so the compression
//     *level* is the lever, not the chunk size.
//   - LZ4 data_type = CHAR (single-pass, no level knob).
//   - nvCOMP Zstd has NO hardware-engine path (DE returns status 10); for a Zstd
//     CUDA-backend baseline use the onpair-chunk-bench at level 3 (not -10).
//
// Build (CUDA >= 12.8, nvcomp SDK under target/.../nvcomp-sdk):
//   SDK=$(find target -path '*nvcomp-sdk' -type d | head -1)
//   nvcc -O3 -arch=native nvcomp_hw_bench.cu -o nvbench \
//     -I"$SDK/include" -L"$SDK/lib" -lnvcomp -lcudart
//   LD_LIBRARY_PATH="$SDK/lib" ./nvbench <file> [chunk_bytes] [deflate_algo]
// Input <file> = raw concatenated column bytes (dump with pyarrow).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <nvcomp/deflate.h>
#include <nvcomp/lz4.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define NK(x) do{ nvcompStatus_t s=(x); if(s!=nvcompSuccess){ fprintf(stderr,"nvcomp %s:%d status=%d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

static size_t CHUNK = 262144; // good-baseline chunk for the DE (overridable via argv[2])

template<class CompOpts, class DecompOpts>
void run(const char* name, std::vector<unsigned char>& host, CompOpts copts, DecompOpts dopts,
         nvcompStatus_t(*compTemp)(size_t,size_t,CompOpts,size_t*,size_t),
         nvcompStatus_t(*compMax)(size_t,CompOpts,size_t*),
         nvcompStatus_t(*compAsync)(const void* const*,const size_t*,size_t,size_t,void*,size_t,void* const*,size_t*,CompOpts,nvcompStatus_t*,cudaStream_t),
         nvcompStatus_t(*decompTemp)(size_t,size_t,DecompOpts,size_t*,size_t),
         nvcompStatus_t(*decompAsync)(const void* const*,const size_t*,const size_t*,size_t*,size_t,void* const,size_t,void* const*,DecompOpts,nvcompStatus_t*,cudaStream_t))
{
    size_t N = host.size();
    size_t num = (N + CHUNK - 1) / CHUNK;
    cudaStream_t stream; CK(cudaStreamCreate(&stream));

    // ---- upload uncompressed, build chunk ptr/size arrays ----
    unsigned char* d_in; CK(cudaMalloc(&d_in, N)); CK(cudaMemcpy(d_in, host.data(), N, cudaMemcpyHostToDevice));
    std::vector<void*> h_inptr(num); std::vector<size_t> h_insz(num);
    for(size_t i=0;i<num;i++){ h_inptr[i]=d_in+i*CHUNK; h_insz[i]=(i+1<num)?CHUNK:(N-i*CHUNK); }
    void** d_inptr; size_t* d_insz;
    CK(cudaMalloc(&d_inptr,num*sizeof(void*))); CK(cudaMemcpy(d_inptr,h_inptr.data(),num*sizeof(void*),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_insz,num*sizeof(size_t))); CK(cudaMemcpy(d_insz,h_insz.data(),num*sizeof(size_t),cudaMemcpyHostToDevice));

    // ---- compress ----
    size_t ctemp=0; NK(compTemp(num,CHUNK,copts,&ctemp,N));
    size_t maxout=0; NK(compMax(CHUNK,copts,&maxout));
    void* d_ctemp=nullptr; if(ctemp) CK(cudaMalloc(&d_ctemp,ctemp));
    unsigned char* d_cbuf; CK(cudaMalloc(&d_cbuf,num*maxout));
    std::vector<void*> h_cptr(num); for(size_t i=0;i<num;i++) h_cptr[i]=d_cbuf+i*maxout;
    void** d_cptr; CK(cudaMalloc(&d_cptr,num*sizeof(void*))); CK(cudaMemcpy(d_cptr,h_cptr.data(),num*sizeof(void*),cudaMemcpyHostToDevice));
    size_t* d_csz; CK(cudaMalloc(&d_csz,num*sizeof(size_t)));
    nvcompStatus_t* d_st; CK(cudaMalloc(&d_st,num*sizeof(nvcompStatus_t)));
    for(int w=0;w<2;w++) NK(compAsync(d_inptr,d_insz,CHUNK,num,d_ctemp,ctemp,d_cptr,d_csz,copts,d_st,stream));
    CK(cudaStreamSynchronize(stream));
    // time compression (encode throughput over uncompressed bytes)
    cudaEvent_t ca,cb; CK(cudaEventCreate(&ca)); CK(cudaEventCreate(&cb));
    int citers=20; CK(cudaEventRecord(ca,stream));
    for(int i=0;i<citers;i++) NK(compAsync(d_inptr,d_insz,CHUNK,num,d_ctemp,ctemp,d_cptr,d_csz,copts,d_st,stream));
    CK(cudaEventRecord(cb,stream)); CK(cudaEventSynchronize(cb));
    float cms=0; CK(cudaEventElapsedTime(&cms,ca,cb)); cms/=citers;
    double enc_gibs=(double)N/(cms/1e3)/(1024.0*1024*1024);
    std::vector<size_t> h_csz(num); CK(cudaMemcpy(h_csz.data(),d_csz,num*sizeof(size_t),cudaMemcpyDeviceToHost));
    size_t ctot=0; for(size_t i=0;i<num;i++) ctot+=h_csz[i];

    // ---- decompress on HARDWARE engine ----
    size_t dtemp=0; nvcompStatus_t ds=decompTemp(num,CHUNK,dopts,&dtemp,N);
    if(ds!=nvcompSuccess){ printf("%-8s HW decompress GetTempSize status=%d (UNSUPPORTED)\n",name,(int)ds); return; }
    void* d_dtemp=nullptr; if(dtemp) CK(cudaMalloc(&d_dtemp,dtemp));
    unsigned char* d_out; CK(cudaMalloc(&d_out,N));
    std::vector<void*> h_optr(num); for(size_t i=0;i<num;i++) h_optr[i]=d_out+i*CHUNK;
    void** d_optr; CK(cudaMalloc(&d_optr,num*sizeof(void*))); CK(cudaMemcpy(d_optr,h_optr.data(),num*sizeof(void*),cudaMemcpyHostToDevice));
    size_t* d_obufsz; CK(cudaMalloc(&d_obufsz,num*sizeof(size_t))); CK(cudaMemcpy(d_obufsz,d_insz,num*sizeof(size_t),cudaMemcpyDeviceToDevice));
    size_t* d_actual; CK(cudaMalloc(&d_actual,num*sizeof(size_t)));

    for(int w=0;w<3;w++){ NK(decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream)); }
    CK(cudaStreamSynchronize(stream));

    // validate
    std::vector<unsigned char> back(N); CK(cudaMemcpy(back.data(),d_out,N,cudaMemcpyDeviceToHost));
    bool ok = memcmp(back.data(),host.data(),N)==0;

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    int iters=100; CK(cudaEventRecord(a,stream));
    for(int i=0;i<iters;i++){ NK(decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream)); }
    CK(cudaEventRecord(b,stream)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); ms/=iters;
    double gibs = (double)N/(ms/1e3)/ (1024.0*1024*1024);
    double gbs  = (double)N/(ms/1e3)/ 1e9;
    printf("%-13s ratio=%.2fx  compress=%6.1f GiB/s  decode=%6.1f GiB/s (%.0f GB/s)  valid=%s\n",
           name, (double)N/ctot, enc_gibs, gibs, gbs, ok?"YES":"NO");
}

int main(int argc, char** argv){
    const char* path = argc>1?argv[1]:"/tmp/l_comment.bin";
    if(argc>2) CHUNK = (size_t)atol(argv[2]);
    FILE* f=fopen(path,"rb"); if(!f){ perror("open"); return 1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<unsigned char> host(sz); fread(host.data(),1,sz,f); fclose(f);
    printf("input: %s  %.1f MiB  (%zu chunks of %zu B)\n", path, sz/1048576.0, (size_t)((sz+CHUNK-1)/CHUNK), CHUNK);
    CK(cudaSetDevice(0));

    // Two presets per HW codec: "hi" = max compression ratio, "fast" = best
    // (de)compression throughput. Deflate exposes a level (algo 0..5): algo=5 is
    // max ratio, algo=0 is "entropy-only, symmetric comp/decomp performance".
    // LZ4 has no level (single-pass) — it is inherently the throughput option.
    nvcompBatchedDeflateDecompressOpts_t ddo = nvcompBatchedDeflateDecompressDefaultOpts;
    ddo.backend = NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
    int deflate_algos[2] = {5, 0};            // hi, fast
    const char* deflate_names[2] = {"DEFLATE-hi","DEFLATE-fast"};
    if(argc>3){ deflate_algos[0]=atoi(argv[3]); deflate_names[0]="DEFLATE-custom"; deflate_algos[1]=-1; }
    for(int p=0;p<2;p++){
        if(deflate_algos[p]<0) continue;
        nvcompBatchedDeflateCompressOpts_t dco = nvcompBatchedDeflateCompressDefaultOpts;
        dco.algorithm = deflate_algos[p];
        run<nvcompBatchedDeflateCompressOpts_t,nvcompBatchedDeflateDecompressOpts_t>(
            deflate_names[p], host, dco, ddo,
            nvcompBatchedDeflateCompressGetTempSizeAsync, nvcompBatchedDeflateCompressGetMaxOutputChunkSize,
            nvcompBatchedDeflateCompressAsync, nvcompBatchedDeflateDecompressGetTempSizeAsync,
            nvcompBatchedDeflateDecompressAsync);
    }

    nvcompBatchedLZ4DecompressOpts_t ldo = nvcompBatchedLZ4DecompressDefaultOpts;
    ldo.backend = NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
    run<nvcompBatchedLZ4CompressOpts_t,nvcompBatchedLZ4DecompressOpts_t>(
        "LZ4", host, nvcompBatchedLZ4CompressDefaultOpts, ldo,
        nvcompBatchedLZ4CompressGetTempSizeAsync, nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
        nvcompBatchedLZ4CompressAsync, nvcompBatchedLZ4DecompressGetTempSizeAsync,
        nvcompBatchedLZ4DecompressAsync);
    return 0;
}
