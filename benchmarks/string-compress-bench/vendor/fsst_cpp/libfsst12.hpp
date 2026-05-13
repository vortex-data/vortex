// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// 
// Copyright 2018-2019, CWI, TU Munich
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files   
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,   
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is   
// furnished to do so, subject to the following conditions:
// 
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
//                 
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst 
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

using namespace std;

#include "fsst12.h" // the official FSST API -- also usable by C mortals

namespace libfsst {
/* workhorse type for string and buffer lengths: 64-bits on 64-bits platforms and 32-bits on 32-bits platforms */
typedef unsigned long ulong; 

/* unsigned integers */
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
} // namespace libfsst

#define FSST_ENDIAN_MARKER ((u64) 1)
#define FSST_VERSION_20190218 20190218
#define FSST_VERSION ((u64) FSST_VERSION_20190218)

// "symbols" are character sequences (up to 8 bytes)
// A symbol is compressed into a "code" of, 1.5 bytes (12 bits)
#define FSST_CODE_MAX 4096
#define FSST_CODE_MASK      ((u16) (FSST_CODE_MAX-1)) 

namespace libfsst {
inline uint64_t fsst_unaligned_load(u8 const* V) {
    uint64_t Ret;
    memcpy(&Ret, V, sizeof(uint64_t)); // compiler will generate efficient code (unaligned load, where possible)
    return Ret;
}

struct Symbol {
   static const unsigned maxLength = 8;

   // gcl = u32 garbageBits:16,code:12,length:4 -- but we avoid exposing this bit-field notation
   u32 gcl;  // use a single u32 to be sure "code" is accessed with one load and can be compared with one comparison
   mutable u32 gain; // mutable because gain value should be ignored in find() on unordered_set of Symbols

   // the byte sequence that this symbol stands for
   u8 symbol[maxLength]; 

   Symbol() : gcl(0) {}

   explicit Symbol(u8 c, u16 code) : gcl((1<<28)|(code<<16)|7) { *(u64*) symbol = c; } // single-char symbol
   explicit Symbol(const char* input, u32 len) {
      if (len < 8) {
         *(u64*) symbol = 0;
         for(u32 i=0; i<len; i++) symbol[i] = input[i];
      } else {
         len = 8;
         *(u64*) symbol = *(u64*) input;
      }
      set_code_len(FSST_CODE_MASK, len);
   }
   explicit Symbol(const char* begin, const char* end) : Symbol(begin, end-begin) {}
   explicit Symbol(const u8* begin, const u8* end) : Symbol((const char*)begin, end-begin) {}
   void set_code_len(u32 code, u32 len) { gcl = (len<<28)|(code<<16)|((8-len)*8); }

   u8 length() const { return gcl >> 28; }
   u16 code() const { return (gcl >> 16) & FSST_CODE_MASK; }
   u8 garbageBits() const { return gcl; }

   u8 first() const { return 0xFF & *(u64*) symbol; }
   u16 first2() const { assert(length() > 1); return (0xFFFF & *(u64*) symbol); }

#define FSST_HASH_LOG2SIZE 14
#define FSST_HASH_SHIFT 15 
#define FSST_HASH_PRIME1 2971215073LL
#define FSST_HASH(w) (((w)*FSST_HASH_PRIME1)^(((w)*FSST_HASH_PRIME1)>>13))
   ulong hash() const { uint v0 = 0xFFFFFFFF & *(ulong*) symbol; return FSST_HASH(v0); }

   bool operator==(const Symbol& other) const { return *(u64*) symbol == *(u64*) other.symbol && length() == other.length(); }
};

// during search for the best dictionary, we probe both (in this order, first wins):  
// - Symbol hashtable[8192] (keyed by the next four bytes, for s.length>2 -- certain 4-byte sequences will map to the same 3-byte symbol), 
// - u16 shortCodes[65536] array at the position of the next two-byte pattern (s.length==2) and 
// this search will yield a u16 code, it points into Symbol symbols[4096].
// you always find a hit, because the lowest 256 codes are all single-byte symbols

// in the hash table, the gcl field contains (low-to-high) garbageBits:16,code:12,length:4 
#define FSST_GCL_FREE ((15<<28)|(((u32)FSST_CODE_MASK)<<16)) // high bits of gcl (len=15,code=FSST_CODE_MASK) indicates free bucket

// garbageBits is (8-length)*8, which is the amount of high bits to zero in the input word before comparing with the hashtable key
//             ..it could of course be computed from len during lookup, but storing it precomputed in some loose bits is faster
//
// the gain field is only used in the symbol queue that sorts symbols on gain

struct SymbolMap {
   static const u32 hashTabSize = 1<<FSST_HASH_LOG2SIZE; // smallest size that incurs no precision loss

   // lookup table using the next two bytes (65536 codes), or just the next single byte
   u16 shortCodes[65536]; // shortCode[X] contains code for 2-byte symbol, contains 1-byte code X&255 if there is no 2-byte symbol

   // 'symbols' is the current symbol  table symbol[code].symbol is the max 8-byte 'symbol' for single-byte 'code'
   Symbol symbols[4096];  

   // replicate long symbols in hashTab (avoid indirection). 
   Symbol hashTab[hashTabSize]; // used for all symbols of 3 and more bytes

   u32 symbolCount;       // amount of symbols in the map (max 4096)
   bool zeroTerminated;   // whether we are expecting zero-terminated strings (we then also produce zero-terminated compressed strings)
   u16 lenHisto[8];        // lenHisto[x] is the amount of symbols of byte-length (x+1) in this SymbolMap

   SymbolMap() : symbolCount(256), zeroTerminated(false) {
      // stuff done once at startup
      Symbol unused = Symbol(0,FSST_CODE_MASK); // single-char symbol, exception code
      for (u32 i=0; i<256; i++) {
         symbols[i] = Symbol((u8)i,i); // single-byte symbol
      }
      for (u32 i=256; i<4096; i++) {
         symbols[i] = unused; // all other symbols are unused.
      }
      // stuff done when re-using a symbolmap during the search for the best map
      clear(); // clears the arrays (hortCodes and hashTab) and histo
   }

   void clear() {
      Symbol s;
      s.gcl = FSST_GCL_FREE; //marks empty in hashtab
      s.gain = 0; 
      for(u32 i=0; i<hashTabSize; i++)
         hashTab[i] = s;
      for(u32 i=0; i<65536; i++)
         shortCodes[i] = 4096 | (i & 255); // single-byte symbol
      memset(lenHisto, 0, sizeof(lenHisto)); // all unused
      lenHisto[0] = symbolCount = 256; // no need to clean symbols[] as no symbols are used
   }
 
   u32 load() {
      u32 ret = 0;
      for(u32 i=0; i<hashTabSize; i++)
         ret += (hashTab[i].gcl < FSST_GCL_FREE);
      return ret;
   }

   bool hashInsert(Symbol s) {
      u32 idx = s.hash() & (hashTabSize-1);
      bool taken = (hashTab[idx].gcl < FSST_GCL_FREE);
      if (taken) return false; // collision in hash table
      hashTab[idx].gcl = s.gcl;
      hashTab[idx].gain = 0;
      *(u64*) hashTab[idx].symbol = (*(u64*) s.symbol) & (0xFFFFFFFFFFFFFFFF >> (u8) s.gcl);
      return true;
   }
   bool add(Symbol s) {
      assert(symbolCount < 4096);
      u32 len = s.length();
      assert(len > 1);
      s.set_code_len(symbolCount, len);
      if (len == 2) {
         assert(shortCodes[s.first2()] == 4096 + s.first()); // cannot be in use
         shortCodes[s.first2()] = 8192 + symbolCount; // 8192 = (len == 2) << 12
      } else if (!hashInsert(s)) {
         return false;
      }
      symbols[symbolCount++] = s;
      lenHisto[len-1]++;
      return true;
   }
   /// Find symbol in hash table, return code
   u16 hashFind(Symbol s) const {
      ulong idx = s.hash() & (hashTabSize-1);
      if (hashTab[idx].gcl <= s.gcl && 
          *(u64*) hashTab[idx].symbol == (*(u64*) s.symbol & (0xFFFFFFFFFFFFFFFF >> ((u8) hashTab[idx].gcl)))) 
         return (hashTab[idx].gcl>>16); // matched a long symbol 
      return 0;
   }
   /// Find longest expansion, return code
   u16 findExpansion(Symbol s) const {
      if (s.length() == 1) { 
	return 4096 + s.first();
      }
      u16 ret = hashFind(s);
      return ret?ret:shortCodes[s.first2()];
   }
};


#if 0 //def NONOPT_FSST
struct Counters {
   u16 count1[FSST_CODE_MAX];   // array to count frequency of symbols as they occur in the sample 
   u16 count2[FSST_CODE_MAX][FSST_CODE_MAX]; // array to count subsequent combinations of two symbols in the sample 

   void count1Set(u32 pos1, u16 val) { 
      count1[pos1] = val;
   }
   void count1Inc(u32 pos1) { 
      count1[pos1]++;
   }
   void count2Inc(u32 pos1, u32 pos2) {  
      count2[pos1][pos2]++;
   }
   u32 count1GetNext(u32 &pos1) { 
      return count1[pos1];
   }
   u32 count2GetNext(u32 pos1, u32 &pos2) { 
      return count2[pos1][pos2];
   }
   void backup1(u8 *buf) {
      memcpy(buf, count1, FSST_CODE_MAX*sizeof(u16));
   }
   void restore1(u8 *buf) {
      memcpy(count1, buf, FSST_CODE_MAX*sizeof(u16));
   }
};
#else
// we keep two counters count1[pos] and count2[pos1][pos2] of resp 16 and 12-bits. Both are split into two columns for performance reasons
// first reason is to make the column we update the most during symbolTable construction (the low bits) thinner, thus reducing CPU cache pressure.
// second reason is that when scanning the array, after seeing a 64-bits 0 in the high bits column, we can quickly skip over many codes (15 or 7)
struct Counters {
   // high arrays come before low arrays, because our GetNext() methods may overrun their 64-bits reads a few bytes
   u8 count1High[FSST_CODE_MAX];   // array to count frequency of symbols as they occur in the sample (16-bits)
   u8 count1Low[FSST_CODE_MAX];    // it is split in a low and high byte: cnt = count1High*256 + count1Low
   u8 count2High[FSST_CODE_MAX][FSST_CODE_MAX/2]; // array to count subsequent combinations of two symbols in the sample (12-bits: 8-bits low, 4-bits high)
   u8 count2Low[FSST_CODE_MAX][FSST_CODE_MAX];    // its value is (count2High*256+count2Low) -- but high is 4-bits (we put two numbers in one, hence /2)
   // 385KB  -- but hot area likely just 10 + 30*4 = 130 cache lines (=8KB)
   
   void count1Set(u32 pos1, u16 val) { 
      count1Low[pos1] = val&255;
      count1High[pos1] = val>>8;
   }
   void count1Inc(u32 pos1) { 
      if (!count1Low[pos1]++) // increment high early (when low==0, not when low==255). This means (high > 0) <=> (cnt > 0)
         count1High[pos1]++; //(0,0)->(1,1)->..->(255,1)->(0,1)->(1,2)->(2,2)->(3,2)..(255,2)->(0,2)->(1,3)->(2,3)...
   }
   void count2Inc(u32 pos1, u32 pos2) {  
       if (!count2Low[pos1][pos2]++) // increment high early (when low==0, not when low==255). This means (high > 0) <=> (cnt > 0)
          // inc 4-bits high counter with 1<<0 (1) or 1<<4 (16) -- depending on whether pos2 is even or odd, repectively
          count2High[pos1][(pos2)>>1] += 1 << (((pos2)&1)<<2); // we take our chances with overflow.. (4K maxval, on a 8K sample)
   }
   u32 count1GetNext(u32 &pos1) { // note: we will advance pos1 to the next nonzero counter in register range
      // read 16-bits single symbol counter, split into two 8-bits numbers (count1Low, count1High), while skipping over zeros
      u64 high = *(u64*) &count1High[pos1]; // note: this reads 8 subsequent counters [pos1..pos1+7]

      u32 zero = high?(__builtin_ctzl(high)>>3):7; // number of zero bytes
      high = (high >> (zero << 3)) & 255; // advance to nonzero counter
      if (((pos1 += zero) >= FSST_CODE_MAX) || !high) // SKIP! advance pos2
         return 0; // all zero

      u64 low = count1Low[pos1];
      if (low) high--; // high is incremented early and low late, so decrement high (unless low==0)
      return (high << 8) + low;
   }
   u32 count2GetNext(u32 pos1, u32 &pos2) { // note: we will advance pos2 to the next nonzero counter in register range
      // read 12-bits pairwise symbol counter, split into low 8-bits and high 4-bits number while skipping over zeros
      u64 high = *(u64*) &count2High[pos1][pos2>>1]; // note: this reads 16 subsequent counters [pos2..pos2+15]
      high >>= (pos2&1) << 2; // odd pos2: ignore the lowest 4 bits & we see only 15 counters

      u32 zero = high?(__builtin_ctzl(high)>>2):(15-(pos2&1)); // number of zero 4-bits counters
      high = (high >> (zero << 2)) & 15;  // advance to nonzero counter
      if (((pos2 += zero) >= FSST_CODE_MAX) || !high) // SKIP! advance pos2
         return 0; // all zero

      u64 low = count2Low[pos1][pos2];
      if (low) high--; // high is incremented early and low late, so decrement high (unless low==0)
      return (high << 8) + low;
   }
   void backup1(u8 *buf) {
      memcpy(buf, count1High, FSST_CODE_MAX);
      memcpy(buf+FSST_CODE_MAX, count1Low, FSST_CODE_MAX);
   }
   void restore1(u8 *buf) {
      memcpy(count1High, buf, FSST_CODE_MAX);
      memcpy(count1Low, buf+FSST_CODE_MAX, FSST_CODE_MAX);
   }
}; 
#endif

// an encoder is a symbolmap plus some bufferspace, needed during map construction as well as compression 
struct Encoder {
   shared_ptr<SymbolMap> symbolMap; // symbols, plus metadata and data structures for quick compression (shortCode,hashTab, etc)
   union {
      Counters counters;     // for counting symbol occurences during map construction
   };
};

// C++ fsst-compress function with some more control of how the compression happens (algorithm flavor, simd unroll degree)
ulong compressImpl(Encoder *encoder, ulong n, ulong lenIn[], u8 *strIn[], ulong size, u8 * output, ulong *lenOut, u8 *strOut[], bool noSuffixOpt, bool avoidBranch, int simd);
ulong compressAuto(Encoder *encoder, ulong n, ulong lenIn[], u8 *strIn[], ulong size, u8 * output, ulong *lenOut, u8 *strOut[], int simd);
}  // namespace libfsst
