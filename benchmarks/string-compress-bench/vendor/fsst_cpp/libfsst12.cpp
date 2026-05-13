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
#include "libfsst12.hpp"
#include <math.h>
#include <string.h>

namespace libfsst {
Symbol concat(Symbol a, Symbol b) {
   Symbol s;
   u32 length = min(8, a.length()+b.length());
   s.set_code_len(FSST_CODE_MASK, length);
   *(u64*) s.symbol = ((*(u64*) b.symbol) << (8*a.length())) | *(u64*) a.symbol;
   return s;
}
}  // namespace libfsst

namespace std {
template <>
class hash<libfsst::Symbol> {
   public:
   size_t operator()(const libfsst::Symbol& s) const {
      using namespace libfsst;
      uint64_t k = *(u64*) s.symbol;
      const uint64_t m = 0xc6a4a7935bd1e995;
      const int r = 47;
      uint64_t h = 0x8445d61a4e774912 ^ (8*m);
      k *= m;
      k ^= k >> r;
      k *= m;
      h ^= k;
      h *= m;
      h ^= h >> r;
      h *= m;
      h ^= h >> r;
      return h;
   }
};
}

namespace libfsst {
std::ostream& operator<<(std::ostream& out, const Symbol& s) {
   for (u32 i=0; i<s.length(); i++)
      out << s.symbol[i];
   return out;
}

#define FSST_SAMPLETARGET (1<<17) 
#define FSST_SAMPLEMAXSZ ((long) 2*FSST_SAMPLETARGET) 

SymbolMap *buildSymbolMap(Counters& counters, long sampleParam, vector<ulong>& sample, const ulong len[], const u8* line[]) {
   ulong sampleSize = max(sampleParam, FSST_SAMPLEMAXSZ); // if sampleParam is negative, we need to ignore part of the last line
   SymbolMap *st = new SymbolMap(), *bestMap = new SymbolMap();
   long bestGain = -sampleSize; // worst case (everything exception)
   ulong sampleFrac = 128;

   for(ulong i=0; i<sample.size(); i++) {
      const u8* cur = line[sample[i]];
      if (sampleParam < 0 && i+1 == sample.size())
         cur -= sampleSize; // use only last part of last line (which could be too long for an efficient sample)
   }

   // a random number between 0 and 128
   auto rnd128 = [&](ulong i) { return 1 + (FSST_HASH((i+1)*sampleFrac)&127); };

   // compress sample, and compute (pair-)frequencies
   auto compressCount = [&](SymbolMap *st, Counters &counters) { // returns gain
      long gain = 0;

      for(ulong i=0; i<sample.size(); i++) {
         const u8* cur = line[sample[i]];
         const u8* end = cur + len[sample[i]];

         if (sampleParam < 0 && i+1 == sample.size()) { 
            cur -= sampleParam; // use only last part of last line (which could be too long for an efficient sample)
            if ((end-cur) > 500) end = cur + ((end-cur)*sampleFrac)/128; // shorten long lines to the sample fraction
         } else if (sampleFrac < 128) {
            // in earlier rounds (sampleFrac < 128) we skip data in the sample (reduces overall work ~2x)
            if (rnd128(i) > sampleFrac) continue;
         }

         if (cur < end) {
            u16 pos2 = 0, pos1 = st->findExpansion(Symbol(cur, end));
            cur += pos1 >> 12;
            pos1 &= FSST_CODE_MASK;
            while (true) {
	       const u8 *old = cur;
               counters.count1Inc(pos1);
               if (cur<end-7) {
                  ulong word = fsst_unaligned_load(cur);
                  ulong pos = (u32) word; // key is first 4 bytes!!
                  ulong idx = FSST_HASH(pos)&(st->hashTabSize-1);
                  Symbol s = st->hashTab[idx];
                  pos2 = st->shortCodes[word & 0xFFFF];
                  word &= (0xFFFFFFFFFFFFFFFF >> (u8) s.gcl);
                  if ((s.gcl < FSST_GCL_FREE) && (*(u64*) s.symbol == word)) {
                     pos2 = s.code(); cur += s.length();
                  } else {
                     cur += (pos2 >> 12);
                     pos2 &= FSST_CODE_MASK;
                  }
               } else if (cur==end) {
                  break;
               } else {
                  assert(cur<end);
                  pos2 = st->findExpansion(Symbol(cur, end));
                  cur += pos2 >> 12;
                  pos2 &= FSST_CODE_MASK;
               }

               // compute compressed output size (later divide by 2)
               gain += 2*(cur-old)-3;

               // now count the subsequent two symbols we encode as an extension possibility
               if (sampleFrac < 128) { // no need to count pairs in final round
                  counters.count2Inc(pos1, pos2);
               }
               pos1 = pos2;
            }
         }
      }
      return gain; 
   };

   auto makeMap = [&](SymbolMap *st, Counters &counters) {
      // hashmap of c (needed because we can generate duplicate candidates)
      unordered_set<Symbol> cands;

      auto addOrInc = [&](unordered_set<Symbol> &cands, Symbol s, u32 count) {
         auto it = cands.find(s);
         s.gain = s.length()*count;
         if (it != cands.end()) {
            s.gain += (*it).gain;
            cands.erase(*it);
         }
         cands.insert(s);
      };

      // add candidate symbols based on counted frequency
      for (u32 pos1=0; pos1<st->symbolCount; pos1++) { 
         u32 cnt1 = counters.count1GetNext(pos1); // may advance pos1!!
         if (!cnt1) continue;

         Symbol s1 = st->symbols[pos1];
         if (s1.length() > 1) { // 1-byte symbols are always in the map
            addOrInc(cands, s1, cnt1);
         }

         if (sampleFrac >= 128 || // last round we do not create new (combined) symbols
             s1.length() == Symbol::maxLength) { // symbol cannot be extended
            continue;
         }
         for (u32 pos2=0; pos2<st->symbolCount; pos2++) { 
            u32 cnt2 = counters.count2GetNext(pos1, pos2); // may advance pos2!!
            if (!cnt2) continue;

            // create a new symbol
            Symbol s2 = st->symbols[pos2];
            Symbol s3 = concat(s1, s2);
            addOrInc(cands, s3, cnt2);
         }
      }

      // insert candidates into priority queue (by gain)
      auto cmpGn = [](const Symbol& q1, const Symbol& q2) { return q1.gain < q2.gain; };
      priority_queue<Symbol,vector<Symbol>,decltype(cmpGn)> pq(cmpGn);
      for (auto& q : cands)
         pq.push(q);

      // Create new symbol map using best candidates
      st->clear();
      while (st->symbolCount < 4096 && !pq.empty()) {
         Symbol s = pq.top();
         pq.pop();
         st->add(s);
      }
   };

#ifdef NONOPT_FSST
   for(ulong frac : {127, 127, 127, 127, 127, 127, 127, 127, 127, 128}) {
      sampleFrac = frac;
#else
   for(sampleFrac=14; true; sampleFrac = sampleFrac + 38) {
#endif
      memset(&counters, 0, sizeof(Counters));
      long gain = compressCount(st, counters);
      if (gain >= bestGain) { // a new best solution!
         *bestMap = *st; bestGain = gain;
      } 
      if (sampleFrac >= 128) break; // we do 4 rounds (sampleFrac=14,52,90,128)
      makeMap(st, counters);
   }
   delete st;
   return bestMap;
}

// optimized adaptive *scalar* compression method
static inline ulong compressBulk(SymbolMap &symbolMap, ulong nlines, const ulong lenIn[], const u8* strIn[], ulong size, u8* out, ulong lenOut[], u8* strOut[]) {
   u8 *lim = out + size;
   ulong curLine;
   for(curLine=0; curLine<nlines; curLine++) {
      const u8 *cur = strIn[curLine];
      const u8 *end = cur + lenIn[curLine];
      strOut[curLine] = out;
      while (cur+16 <= end && (lim-out) >= 8) {
         u64 word = fsst_unaligned_load(cur);
         ulong code = symbolMap.shortCodes[word & 0xFFFF];
         ulong pos = (u32) word; // key is first 4 bytes
         ulong idx = FSST_HASH(pos)&(symbolMap.hashTabSize-1);
         Symbol s = symbolMap.hashTab[idx];
         word &= (0xFFFFFFFFFFFFFFFF >> (u8) s.gcl);
         if ((s.gcl < FSST_GCL_FREE) && *(ulong*) s.symbol == word) {
            code = s.gcl >> 16;
         }
         cur += (code >> 12);
         u32 res = code & FSST_CODE_MASK;
         word = fsst_unaligned_load(cur);
         code = symbolMap.shortCodes[word & 0xFFFF];
         pos = (u32) word; // key is first 4 bytes
         idx = FSST_HASH(pos)&(symbolMap.hashTabSize-1);
         s = symbolMap.hashTab[idx];
         word &= (0xFFFFFFFFFFFFFFFF >> (u8) s.gcl);
         if ((s.gcl < FSST_GCL_FREE) && *(ulong*) s.symbol == word) {
           code = s.gcl >> 16;
         }
         cur += (code >> 12);
         res |= (code&FSST_CODE_MASK) << 12;
         memcpy(out, &res, sizeof(u32));
         out += 3; 
      }
      while (cur < end) {
         ulong code = symbolMap.findExpansion(Symbol(cur, end));
         u32 res = (code&FSST_CODE_MASK);
         if (out+8 > lim) {
             return curLine; // u32 write would be out of bounds (out of output memory) 
         }
         cur += code >> 12;
         if (cur >= end) {
            memcpy(out, &res, sizeof(u32)); 
			out += 2;
            break;
         }
         code = symbolMap.findExpansion(Symbol(cur, end));
         res |= (code&FSST_CODE_MASK) << 12;
         cur += code >> 12;
         memcpy(out, &res, sizeof(u32));
	 out += 3;
      } 
      lenOut[curLine] = out - strOut[curLine];
   } 
   return curLine;
}

long makeSample(vector<ulong> &sample, ulong nlines, const ulong len[]) {
   ulong i, sampleRnd = 1, sampleProb = 256, sampleSize = 0, totSize = 0;
   ulong sampleTarget = FSST_SAMPLETARGET;

   for(i=0; i<nlines; i++) 
      totSize += len[i];

   if (totSize > FSST_SAMPLETARGET) {
      // if the batch is larger than the sampletarget, sample this fraction  
      sampleProb = max(((ulong) 4),(256*sampleTarget) / totSize);
   } else {
      // too little data. But ok, do not include lines multiple times, just use everything once
      sampleTarget = totSize; // sampleProb will be 256/256 (aka 100%) 
   } 
   do {
      // if nlines is very large and strings are small (8, so we need 4K lines), we still expect 4K*256/4 iterations total worst case
      for(i=0; i<nlines; i++) { 
         // cheaply draw a random number to select (or not) each line
         sampleRnd = FSST_HASH(sampleRnd);
         if ((sampleRnd&255) < sampleProb) {
            sample.push_back(i);
            sampleSize += len[i];
            if (sampleSize >= sampleTarget) // enough? 
               i = nlines; // break out of both loops; 
         }
      }
      sampleProb *= 4; //accelerate the selection process at expense of front-bias (4,16,64,256: 4 passes max)
   } while(i <= nlines); // basically continue until we have enough

   // if the last line (only line?) is excessively long, return a negative samplesize (the amount of front bytes to skip)
   long sampleLong = (long) sampleSize;
   assert(sampleLong > 0);
   return (sampleLong < FSST_SAMPLEMAXSZ)?sampleLong:FSST_SAMPLEMAXSZ-sampleLong; 
}
}  // namespace libfsst

using namespace libfsst; 
extern "C" fsst_encoder_t* fsst_create(ulong n, const ulong lenIn[], const u8 *strIn[], int dummy) {
   vector<ulong> sample;
   (void) dummy;
   long sampleSize = makeSample(sample, n?n:1, lenIn); // careful handling of input to get a right-size and representative sample
   Encoder *encoder = new Encoder();
   encoder->symbolMap = shared_ptr<SymbolMap>(buildSymbolMap(encoder->counters, sampleSize, sample, lenIn, strIn));
   return (fsst_encoder_t*) encoder;
}

/* create another encoder instance, necessary to do multi-threaded encoding using the same dictionary */
extern "C" fsst_encoder_t* fsst_duplicate(fsst_encoder_t *encoder) {
  Encoder *e = new Encoder();
   e->symbolMap = ((Encoder*)encoder)->symbolMap; // it is a shared_ptr
   return (fsst_encoder_t*) e;
}

// export a dictionary in compact format. 
extern "C" u32 fsst_export(fsst_encoder_t *encoder, u8 *buf) {
  Encoder *e = (Encoder*) encoder;
   // In ->version there is a versionnr, but we hide also suffixLim/terminator/symbolCount there.
   // This is sufficient in principle to *reconstruct* a fsst_encoder_t from a fsst_decoder_t
   // (such functionality could be useful to append compressed data to an existing block).
   //
   // However, the hash function in the encoder hash table is endian-sensitive, and given its
   // 'lossy perfect' hashing scheme is *unable* to contain other-endian-produced symbol tables.
   // Doing a endian-conversion during hashing will be slow and self-defeating.
   //
   // Overall, we could support reconstructing an encoder for incremental compression, but 
   // should enforce equal-endianness. Bit of a bummer. Not going there now.
   // 
   // The version field is now there just for future-proofness, but not used yet
   
   // version allows keeping track of fsst versions, track endianness, and encoder reconstruction
   u64 version = (FSST_VERSION << 32) | FSST_ENDIAN_MARKER; // least significant byte is nonzero

   /* do not assume unaligned reads here */
   memcpy(buf, &version, 8);
   memcpy(buf+8, e->symbolMap->lenHisto, 16); // serialize the lenHisto
   u32 pos = 24;

   // emit only the used bytes of the symbols 
   for(u32 i = 0; i < e->symbolMap->symbolCount; i++) {
      buf[pos++] = e->symbolMap->symbols[i].length();
      for(u32 j = 0; j < e->symbolMap->symbols[i].length(); j++) {
         buf[pos++] = ((u8*) &e->symbolMap->symbols[i].symbol)[j]; // serialize used symbol bytes
      }
   }
   return pos; // length of what was serialized
}

#define FSST_CORRUPT 32774747032022883 /* 7-byte number in little endian containing "corrupt" */

extern "C" u32 fsst_import(fsst_decoder_t *decoder, u8 *buf) {
   u64 version = 0, symbolCount = 0;
   u32 pos = 24;
   u16 lenHisto[8];

   // version field (first 8 bytes) is now there just for future-proofness, unused still (skipped)
   memcpy(&version, buf, 8);
   if ((version>>32) != FSST_VERSION) return 0;
   memcpy(lenHisto, buf+8, 16);

   for(u32 i=0; i<8; i++) 
     symbolCount += lenHisto[i]; 

   for(u32 i = 0; i < symbolCount; i++) {
      u32 len = decoder->len[i] = buf[pos++];
      for(u32 j = 0; j < len; j++) {
        ((u8*) &decoder->symbol[i])[j] = buf[pos++];
      }
   }
   // fill unused symbols with text "corrupt". Gives a chance to detect corrupted code sequences (if there are unused symbols).
   while(symbolCount<4096) {
       decoder->symbol[symbolCount] = FSST_CORRUPT;    
       decoder->len[symbolCount++] = 8;
   }
   return pos;
}

namespace libfsst {
// runtime check for simd
inline ulong _compressImpl(Encoder *e, ulong nlines, const ulong lenIn[], const u8 *strIn[], ulong size, u8 *output, ulong *lenOut, u8 *strOut[], bool noSuffixOpt, bool avoidBranch, int simd) {
   (void) noSuffixOpt;
   (void) avoidBranch;
   (void) simd;
   return compressBulk(*e->symbolMap, nlines, lenIn, strIn, size, output, lenOut, strOut);
}
ulong compressImpl(Encoder *e, ulong nlines, const ulong lenIn[], const u8 *strIn[], ulong size, u8 *output, ulong *lenOut, u8 *strOut[], bool noSuffixOpt, bool avoidBranch, int simd) {
   return _compressImpl(e, nlines, lenIn, strIn, size, output, lenOut, strOut, noSuffixOpt, avoidBranch, simd);
}

// adaptive choosing of scalar compression method based on symbol length histogram 
inline ulong _compressAuto(Encoder *e, ulong nlines, const ulong lenIn[], const u8 *strIn[], ulong size, u8 *output, ulong *lenOut, u8 *strOut[], int simd) {
   (void) simd;
   return _compressImpl(e, nlines, lenIn, strIn, size, output, lenOut, strOut, false, false, false);
}
ulong compressAuto(Encoder *e, ulong nlines, const ulong lenIn[], const u8 *strIn[], ulong size, u8 *output, ulong *lenOut, u8 *strOut[], int simd) {
   return _compressAuto(e, nlines, lenIn, strIn, size, output, lenOut, strOut, simd);
}
}  // namespace libfsst

using namespace libfsst;
// the main compression function (everything automatic)
extern "C" ulong fsst_compress(fsst_encoder_t *encoder, ulong nlines, const ulong lenIn[], const u8 *strIn[], ulong size, u8 *output, ulong *lenOut, u8 *strOut[]) {
   // to be faster than scalar, simd needs 64 lines or more of length >=12; or fewer lines, but big ones (totLen > 32KB)
   ulong totLen = accumulate(lenIn, lenIn+nlines, 0);
   int simd = totLen > nlines*12 && (nlines > 64 || totLen > (ulong) 1<<15); 
   return _compressAuto((Encoder*) encoder, nlines, lenIn, strIn, size, output, lenOut, strOut, 3*simd);
}

/* deallocate encoder */
extern "C" void fsst_destroy(fsst_encoder_t* encoder) {
  Encoder *e = (Encoder*) encoder; 
   delete e;
}

/* very lazy implementation relying on export and import */
extern "C" fsst_decoder_t fsst_decoder(fsst_encoder_t *encoder) {
   u8 buf[sizeof(fsst_decoder_t)];
   u32 cnt1 = fsst_export(encoder, buf);
   fsst_decoder_t decoder;
   u32 cnt2 = fsst_import(&decoder, buf);
   assert(cnt1 == cnt2); (void) cnt1; (void) cnt2; 
   return decoder;
}
