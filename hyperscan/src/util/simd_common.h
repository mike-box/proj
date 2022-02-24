/*
 * Copyright (c) 2015-2017, Intel Corporation
 * 2020.01 - Use the neon instruction to implement the function of 128-bit operation.
 *           Huawei Technologies Co., Ltd.
 * 2022.02 - Use the C language to implement the function of 128-bit operation.
 *           write by mike meng.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** \file
 * \brief SIMD types and primitive operations.
 */

#ifndef SIMD_COMMON
#define SIMD_COMMON

#include "config.h"
#include "simd_types.h"
#include "ue2common.h"
#include "unaligned.h"
#include "util/arch.h"
#include "util/intrinsics.h"

#include <string.h> // for memcpy

// Define a common assume_aligned using an appropriate compiler built-in, if
// it's available. Note that we need to handle C or C++ compilation.
#ifdef __cplusplus
#ifdef HAVE_CXX_BUILTIN_ASSUME_ALIGNED
#define assume_aligned(x, y) __builtin_assume_aligned((x), (y))
#endif
#else
#ifdef HAVE_CC_BUILTIN_ASSUME_ALIGNED
#define assume_aligned(x, y) __builtin_assume_aligned((x), (y))
#endif
#endif

// Fallback to identity case.
#ifndef assume_aligned
#define assume_aligned(x, y) (x)
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern const char vbs_mask_data[];
#ifdef __cplusplus
}
#endif


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

/* return 1 if a is not 0, else 0 */
static really_inline u8 isnonezero8(u8 a) {
	a = (a&0x55)|((a>>1)&0x55);
	a = (a&0x11)|((a>>2)&0x11);
	a = (a&0x1)|((a>>4)&0x1);
	return a;
}

/* return 1 if a is not 0, else 0 */
static really_inline u16 isnonezero16(u16 a) {
	a = (a&0x5555)|((a>>1)&0x5555);
	a = (a&0x1111)|((a>>2)&0x1111);
	a = (a&0x101)|((a>>4)&0x101);	
	a = (a&0x1)|((a>>8)&0x1);
	return a;
}

/* return 1 if a is not 0, else 0 */
static really_inline u32 isnonezero32(u32 a) {
	a = (a&0x55555555)|((a>>1)&0x55555555);
	a = (a&0x11111111)|((a>>2)&0x11111111);
	a = (a&0x1010101)|((a>>4)&0x1010101);	
	a = (a&0x10001)|((a>>8)&0x10001);	
	a = (a&0x1)|((a>>16)&0x1);
	return a;
}

/* return 1 if a is not 0, else 0 */
static really_inline u64a isnonezero64(u64a a) {
	a = (a&0x5555555555555555)|((a>>1)&0x5555555555555555);
	a = (a&0x1111111111111111)|((a>>2)&0x1111111111111111);
	a = (a&0x101010101010101)|((a>>4)&0x101010101010101);	
	a = (a&0x1000100010001)|((a>>8)&0x1000100010001);	
	a = (a&0x100000001)|((a>>16)&0x100000001);
	a = (a&0x1)|((a>>32)&0x1);
	return a;
}


static really_inline m128 ones128(void) {
    m128 rv;
	memset(&rv, 0xff, 128);
    return rv;
}

static really_inline m128 zeroes128(void) {
    m128 rv;
	memset(&rv, 0, 128);
    return rv;
}

/** \brief Return 1 if a and b are different otherwise 0 */
static really_inline int diff128(m128 a, m128 b) {
	u32 s0 = (a.data_u32[0]^b.data_u32[0]);
	u32 s1 = (a.data_u32[1]^b.data_u32[1]);
	u32 s2 = (a.data_u32[2]^b.data_u32[2]);
	u32 s3 = (a.data_u32[3]^b.data_u32[3]);
	return (s0^s1^s2^s3) == 0 ? 0 : 1;
}

static really_inline int isnonzero128(m128 a) {
    return !!diff128(a, zeroes128());
}

/**
 * "Rich" version of diff128(). Takes two vectors a and b and returns a 4-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich128(m128 a, m128 b) {
    u32 s0 = isnonezero32(a.data_u32[0]^b.data_u32[0]);
	u32 s1 = isnonezero32(a.data_u32[1]^b.data_u32[1]);
	u32 s2 = isnonezero32(a.data_u32[2]^b.data_u32[2]);
	u32 s3 = isnonezero32(a.data_u32[3]^b.data_u32[3]);
	return s0|(s1<<1)|(s2<<2)|(s3<<3);
}

/**
 * "Rich" version of diff128(), 64-bit variant. Takes two vectors a and b and
 * returns a 4-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_128(m128 a, m128 b) {
	u32 d = diffrich128(a, b);
    return (d | (d >> 1)) & 0x5;
}

/*left shift each 64 bit */
static really_really_inline m128 lshift64_m128(m128 a, unsigned b) {
    assert(b <= 63);
	if (unlikely(b == 0)) {
        return a;
    }
    m128 rv;
	rv.data_u64[0] = a.data_u64[0]<<b;
	rv.data_u64[1] = a.data_u64[1]<<b;
    return rv;
}

/*right shift each 64 bit */
static really_really_inline m128 rshift64_m128(m128 a, int imm8) {
    assert(imm8 >= 0 && imm8 <= 63);
    if (unlikely(imm8 == 0)) {
        return a;
    }
    m128 rv;
    rv.data_u64[0] = a.data_u64[0]>>imm8;
	rv.data_u64[1] = a.data_u64[1]>>imm8;
    return rv;
}

/*compare each 1 byte of two m128 number*/
static really_really_inline m128 eq128(m128 a, m128 b) {
    m128 rv;
	rv.data_u64[0] = a.data_u64[0]^b.data_u64[0];
	rv.data_u64[1] = a.data_u64[1]^b.data_u64[1];
	for (int i = 0; i < 16; i++) {
		rv.data_u8[i] = (rv.data_u8[i] == 0 ? 0xff : 0);
	}
	return rv;
}

/* get significant bit of each 8-bit element in a */
static really_really_inline u32 movemask128(m128 a) {
	u32 result = 0;
	u32 s0 = a.data_u32[0]&0x80808080;
	u32 s1 = a.data_u32[1]&0x80808080;
	u32 s2 = a.data_u32[2]&0x80808080;
	u32 s3 = a.data_u32[3]&0x80808080;
	s0 = ((s0>>7)|(s0>>14)|(s0>>21)|(s0>>28))&0xf;
	s1 = ((s1>>7)|(s1>>14)|(s1>>21)|(s1>>28))&0xf;
	s2 = ((s2>>7)|(s2>>14)|(s2>>21)|(s2>>28))&0xf;
	s3 = ((s3>>7)|(s3>>14)|(s3>>21)|(s3>>28))&0xf;
	return s0|(s1<<4)|(s2<<8)|(s3<<12);
}

/* right shift imm8 bytes*/
static really_really_inline m128 rshiftbyte_m128(m128 a, int imm8) {
	assert(imm8 >= 0 && imm8 <= 15);
	if (unlikely(imm8 == 0)) {
		return a;
	}
	m128 rv;
	// high bits shift
	rv.data_u64[0] = (a.data_u64[0]>>(imm8<<3))|(a.data_u64[1]&(~(1<<(imm8<<3))));
	// low bits shift
	rv.data_u64[1] = a.data_u64[1]>>(imm8<<3);	
    return rv;
}

/* left shift imm8 bytes*/
static really_really_inline m128 lshiftbyte_m128(m128 a, int imm8) {
    assert(imm8 >= 0 && imm8 <= 15);
	if (unlikely(imm8 == 0)) {
        return a;
    }
    m128 rv;
	rv.data_u64[0] = a.data_u64[0]<<(imm8<<3);
	rv.data_u64[1] = a.data_u64[1]<<(imm8<<3)|(a.data_u64[0]>>((16-imm8)<<3));
    return rv;
}

/* fill the value with bytes of c */
static really_inline m128 set16x8(u8 c) {
    m128 rv;
	memset(&rv, c, 128);
    return rv;
}

/* fill the value with int of c */
static really_inline m128 set4x32(u32 c) {
    m128 rv;
	rv.data_u32[0] = c;
	rv.data_u32[1] = c;
	rv.data_u32[2] = c;
	rv.data_u32[3] = c;	
    return rv;
}

/* fill the value with long of c */
static really_inline m128 set2x64(u64a c) {
    m128 rv;
	rv.data_u64[0] = c;
	rv.data_u64[1] = c;
    return rv;
}

/* fetch the lower 4 bytes of in */
static really_inline u32 movd(const m128 in) {
    u32 rv;
    rv = in.data_u32[3];
    return rv;
}

/* fetch the lower 8 bytes of in */
static really_inline u64a movq(const m128 in) {
    u64a rv;
    rv = in.data_u64[1];
    return rv;
}

/* another form of movq */
static really_inline m128 load_m128_from_u64a(const u64a *p) {
    m128 rv;
    rv.data_u64[0] = 0;
	rv.data_u64[1] = *p;
    return rv;
}

/*The x86 platform does not perform the lower 2 bit operation.
If the value of imm exceeds 2 bit, a compilation error occurs.*/
static really_inline u32 extract32from128(m128 a, int imm) {
	assert(imm8 >= 0 && imm8 <= 3);
	int c = imm & 0x0003;
    u32 rv = a.data_u32[c];
    return rv;
}

/*The x86 platform does not perform the lower 1 bit operation.
If the value of imm exceeds 1 bit, a compilation error occurs.*/
static really_inline u64a extract64from128(m128 a, int imm) {
	assert(imm8 >= 0 && imm8 <= 1);
	int c = imm & 0x0001;
    u64a rv = a.data_u64[c];
    return rv;
}

#define extractlow64from256(a) movq(a.lo)
#define extractlow32from256(a) movd(a.lo)

/*The x86 platform does not perform the lower 2 bit operation.
If the value of imm exceeds 2 bit, a compilation error occurs.*/
static really_inline u32 extract32from256(m256 a, int imm) {
	assert(imm8 >= 0 && imm8 <= 7);
	int c = imm & 0x0003;
    return (imm >> 2) ? a.hi.data_u32[c] : a.lo.data_u32[c];    
}

/*The x86 platform does not perform the lower 1 bit operation.
If the value of imm exceeds 1 bit, a compilation error occurs.*/
static really_inline u64a extract64from256(m256 a, int imm) {
    assert(imm8 >= 0 && imm8 <= 3);
    int c = imm & 0x0001;
	return (imm >> 1) ? a.hi.data_u64[c] : a.lo.data_u64[c];
}

static really_inline m128 and128(m128 a, m128 b) {
    m128 rv;
	rv.data_u32[0] = a.data_u32[0]&b.data_u32[0];
	rv.data_u32[1] = a.data_u32[1]&b.data_u32[1];
	rv.data_u32[2] = a.data_u32[2]&b.data_u32[2];
	rv.data_u32[3] = a.data_u32[3]&b.data_u32[3];
    return rv;
}

static really_inline m128 not128(m128 a) {
    m128 rv;
    rv.data_u32[0] = ~a.data_u32[0];
	rv.data_u32[1] = ~a.data_u32[1];
	rv.data_u32[2] = ~a.data_u32[2];
	rv.data_u32[3] = ~a.data_u32[3];
    return rv;
}

static really_inline m128 xor128(m128 a, m128 b) {
    m128 rv;
    rv.data_u32[0] = a.data_u32[0]^b.data_u32[0];
	rv.data_u32[1] = a.data_u32[1]^b.data_u32[1];
	rv.data_u32[2] = a.data_u32[2]^b.data_u32[2];
	rv.data_u32[3] = a.data_u32[3]^b.data_u32[3];
    return rv;
}

static really_inline m128 or128(m128 a, m128 b) {
    m128 rv;
    rv.data_u32[0] = a.data_u32[0]^b.data_u32[0];
	rv.data_u32[1] = a.data_u32[1]^b.data_u32[1];
	rv.data_u32[2] = a.data_u32[2]^b.data_u32[2];
	rv.data_u32[3] = a.data_u32[3]^b.data_u32[3];
    return rv;
}

static really_inline m128 andnot128(m128 a, m128 b) {
    m128 rv;
    rv.data_u32[0] = (~a.data_u32[0])&b.data_u32[0];
	rv.data_u32[1] = (~a.data_u32[1])&b.data_u32[1];
	rv.data_u32[2] = (~a.data_u32[2])&b.data_u32[2];
	rv.data_u32[3] = (~a.data_u32[3])&b.data_u32[3];
    return rv;
}

// aligned load
static really_inline m128 load128(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    ptr = assume_aligned(ptr, 16);
    m128 rv;
	memcpy(&rv, ptr, 128);
    return rv;
}

// aligned store
static really_inline void store128(void *ptr, m128 a) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    ptr = assume_aligned(ptr, 16);
	memcpy(ptr, &a, 128);
}

// unaligned load
static really_inline m128 loadu128(const void *ptr) {
    m128 result;
    memcpy(&result, ptr, 128);
    return result;
}

// unaligned store
static really_inline void storeu128(void *ptr, m128 a) {
    memcpy(ptr, &a, 128);
}

// packed unaligned store of first N bytes
static really_inline void storebytes128(void *ptr, m128 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline m128 loadbytes128(const void *ptr, unsigned int n) {
    m128 a = zeroes128();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

#ifdef __cplusplus
extern "C" {
#endif
extern const u8 simd_onebit_masks[];
#ifdef __cplusplus
}
#endif

static really_inline m128 mask1bit128(unsigned int n) {
    assert(n < sizeof(m128) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu128(&simd_onebit_masks[mask_idx]);
}

// switches on bit N in the given vector.
static really_inline void setbit128(m128 *ptr, unsigned int n) {
    *ptr = or128(mask1bit128(n), *ptr);
}

// switches off bit N in the given vector.
static really_inline void clearbit128(m128 *ptr, unsigned int n) {
    *ptr = andnot128(mask1bit128(n), *ptr);
}

// tests bit N in the given vector.
static really_inline char testbit128(m128 val, unsigned int n) {
    const m128 mask = mask1bit128(n);
    return isnonzero128(and128(mask, val));
}

// offset must be an immediate
/*The x86 platform does not perform the lower 8 bit operation.
If the value of imm exceeds 8 bit, a compilation error occurs.*/
static really_inline m128 palignr(m128 a, m128 b, int count) {
    count = count & 0xff;
    if (likely(count < 16)) {
        return extbyte_m128(b, a, count);
    } else if (count < 32) {
        return extbyte_m128(a, zeroes128(), count - 16);
    } else {
        return zeroes128();
    }
}

static really_inline m128 pshufb_m128(m128 a, m128 b) {
    m128 rv;
    for (int i = 0; i < 16; i++) {
		u8 c = b.data_u8[i];
		rv.data_u8[i] = (c&0x80) ? 0 : a.data_u8[(c&0xf)<<3];
	}
    return rv;
}

static really_inline m256 pshufb_m256(m256 a, m256 b) {
    m256 rv;
    rv.lo = pshufb_m128(a.lo, b.lo);
    rv.hi = pshufb_m128(a.hi, b.hi);
    return rv;
}

static really_inline m128 variable_byte_shift_m128(m128 in, s32 amount) {
    assert(amount >= -16 && amount <= 16);
    m128 shift_mask = loadu128(vbs_mask_data + 16 - amount);
    return pshufb_m128(in, shift_mask);
}

static really_inline m128 max_u8_m128(m128 a, m128 b) {
    m128 rv;
    for (int i = 0; i < 16; i++) {
		rv.data_u8[i] = a.data_u8[i] > b.data_u8 ? a.data_u8[i]:b.data_u8[i];
    }
    return rv;
}

static really_inline m128 min_u8_m128(m128 a, m128 b) {
    m128 rv;
    for (int i = 0; i < 16; i++) {
		rv.data_u8[i] = a.data_u8[i] < b.data_u8 ? a.data_u8[i]:b.data_u8[i];
    }
    return rv;
}

static really_inline m128 sadd_u8_m128(m128 a, m128 b) {
    m128 rv;
    for (int i = 0; i < 16; i++) {
		rv.data_u8[i] = a.data_u8[i] + b.data_u8;
    }
    return rv;
}

static really_inline m128 sub_u8_m128(m128 a, m128 b) {
    m128 rv;
    for (int i = 0; i < 16; i++) {
		rv.data_u8[i] = a.data_u8[i] * b.data_u8;
    }
    return rv;
}

static really_inline m128 set64x2(int64_t hi, int64_t lo) {
    m128 rv;
	rv.data_u64[0] = hi;
	rv.data_u64[1] = lo;
    return rv;
}

static really_inline m128 set32x4(int i3, int i2, int i1, int i0) {
    m128 rv;
    rv.data_u32[0] = i3;
	rv.data_u32[1] = i2;
	rv.data_u32[2] = i1;
	rv.data_u32[3] = i0;
    return rv;
}

/****
 **** 256-bit Primitives
 ****/

static really_really_inline m256 lshift64_m256(m256 a, int b) {
    m256 rv = a;
    rv.lo = lshift64_m128(rv.lo, b);
    rv.hi = lshift64_m128(rv.hi, b);
    return rv;
}

static really_inline m256 rshift64_m256(m256 a, int b) {
    m256 rv = a;
    rv.lo = rshift64_m128(rv.lo, b);
    rv.hi = rshift64_m128(rv.hi, b);
    return rv;
}
static really_inline m256 set32x8(u32 in) {
    m256 rv;
    rv.lo = set16x8((u8)in);
    rv.hi = rv.lo;
    return rv;
}

static really_inline m256 eq256(m256 a, m256 b) {
    m256 rv;
    rv.lo = eq128(a.lo, b.lo);
    rv.hi = eq128(a.hi, b.hi);
    return rv;
}

static really_inline u32 movemask256(m256 a) {
    u32 lo_mask = movemask128(a.lo);
    u32 hi_mask = movemask128(a.hi);
    return lo_mask | (hi_mask << 16);
}

static really_inline m256 set2x128(m128 a) {
    m256 rv = {a, a};
    return rv;
}

static really_inline m256 zeroes256(void) {
    m256 rv = {zeroes128(), zeroes128()};
    return rv;
}

static really_inline m256 ones256(void) {
    m256 rv = {ones128(), ones128()};
    return rv;
}

static really_inline m256 and256(m256 a, m256 b) {
    m256 rv;
    rv.lo = and128(a.lo, b.lo);
    rv.hi = and128(a.hi, b.hi);
    return rv;
}

static really_inline m256 or256(m256 a, m256 b) {
    m256 rv;
    rv.lo = or128(a.lo, b.lo);
    rv.hi = or128(a.hi, b.hi);
    return rv;
}

static really_inline m256 xor256(m256 a, m256 b) {
    m256 rv;
    rv.lo = xor128(a.lo, b.lo);
    rv.hi = xor128(a.hi, b.hi);
    return rv;
}

static really_inline m256 not256(m256 a) {
    m256 rv;
    rv.lo = not128(a.lo);
    rv.hi = not128(a.hi);
    return rv;
}

static really_inline m256 andnot256(m256 a, m256 b) {
    m256 rv;
    rv.lo = andnot128(a.lo, b.lo);
    rv.hi = andnot128(a.hi, b.hi);
    return rv;
}

static really_inline int diff256(m256 a, m256 b) {
    return diff128(a.lo, b.lo) || diff128(a.hi, b.hi);
}

static really_inline int isnonzero256(m256 a) {
    return isnonzero128(or128(a.lo, a.hi));
}

/**
 * "Rich" version of diff256(). Takes two vectors a and b and returns an 8-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich256(m256 a, m256 b) {
    static const int8_t __attribute__((aligned(16)))
  	u32 s0 = diffrich128(a.hi, b.hi);
	u32 s1 = diffrich128(a.lo, b.lo);
    return (s0<<4)|(s1);
}

/**
 * "Rich" version of diff256(), 64-bit variant. Takes two vectors a and b and
 * returns an 8-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_256(m256 a, m256 b) {
    u32 d = diffrich256(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m256 load256(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    m256 rv = {load128(ptr), load128((const char *)ptr + 16)};
    return rv;
}

// aligned load  of 128-bit value to low and high part of 256-bit value
static really_inline m256 load2x128(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    m256 rv;
    rv.hi = rv.lo = load128(ptr);
    return rv;
}

static really_inline m256 loadu2x128(const void *ptr) {
    return set2x128(loadu128(ptr));
}

// aligned store
static really_inline void store256(void *ptr, m256 a) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    ptr = assume_aligned(ptr, 16);
    *(m256 *)ptr = a;
}

// unaligned load
static really_inline m256 loadu256(const void *ptr) {
    m256 rv = {loadu128(ptr), loadu128((const char *)ptr + 16)};
    return rv;
}

// unaligned store
static really_inline void storeu256(void *ptr, m256 a) {
    storeu128(ptr, a.lo);
    storeu128((char *)ptr + 16, a.hi);
}

// packed unaligned store of first N bytes
static really_inline void storebytes256(void *ptr, m256 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline m256 loadbytes256(const void *ptr, unsigned int n) {
    m256 a = zeroes256();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

static really_inline m256 mask1bit256(unsigned int n) {
    assert(n < sizeof(m256) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu256(&simd_onebit_masks[mask_idx]);
}

static really_inline m256 set64x4(u64a hi_1, u64a hi_0, u64a lo_1, u64a lo_0) {
    m256 rv;
    rv.hi = set64x2(hi_1, hi_0);
    rv.lo = set64x2(lo_1, lo_0);
    return rv;
}

// switches on bit N in the given vector.
static really_inline void setbit256(m256 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else {
        sub = &ptr->hi;
        n -= 128;
    }
    setbit128(sub, n);
}

// switches off bit N in the given vector.
static really_inline void clearbit256(m256 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else {
        sub = &ptr->hi;
        n -= 128;
    }
    clearbit128(sub, n);
}

// tests bit N in the given vector.
static really_inline char testbit256(m256 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo;
    } else {
        sub = val.hi;
        n -= 128;
    }
    return testbit128(sub, n);
}

static really_really_inline m128 movdq_hi(m256 x) { return x.hi; }

static really_really_inline m128 movdq_lo(m256 x) { return x.lo; }

static really_inline m256 combine2x128(m128 hi, m128 lo) {
    m256 rv = {lo, hi};
    return rv;
}

/****
 **** 384-bit Primitives
 ****/

static really_inline m384 and384(m384 a, m384 b) {
    m384 rv;
    rv.lo = and128(a.lo, b.lo);
    rv.mid = and128(a.mid, b.mid);
    rv.hi = and128(a.hi, b.hi);
    return rv;
}

static really_inline m384 or384(m384 a, m384 b) {
    m384 rv;
    rv.lo = or128(a.lo, b.lo);
    rv.mid = or128(a.mid, b.mid);
    rv.hi = or128(a.hi, b.hi);
    return rv;
}

static really_inline m384 xor384(m384 a, m384 b) {
    m384 rv;
    rv.lo = xor128(a.lo, b.lo);
    rv.mid = xor128(a.mid, b.mid);
    rv.hi = xor128(a.hi, b.hi);
    return rv;
}
static really_inline m384 not384(m384 a) {
    m384 rv;
    rv.lo = not128(a.lo);
    rv.mid = not128(a.mid);
    rv.hi = not128(a.hi);
    return rv;
}
static really_inline m384 andnot384(m384 a, m384 b) {
    m384 rv;
    rv.lo = andnot128(a.lo, b.lo);
    rv.mid = andnot128(a.mid, b.mid);
    rv.hi = andnot128(a.hi, b.hi);
    return rv;
}

static really_really_inline m384 lshift64_m384(m384 a, unsigned b) {
    m384 rv;
    rv.lo = lshift64_m128(a.lo, b);
    rv.mid = lshift64_m128(a.mid, b);
    rv.hi = lshift64_m128(a.hi, b);
    return rv;
}

static really_inline m384 zeroes384(void) {
    m384 rv = {zeroes128(), zeroes128(), zeroes128()};
    return rv;
}

static really_inline m384 ones384(void) {
    m384 rv = {ones128(), ones128(), ones128()};
    return rv;
}

static really_inline int diff384(m384 a, m384 b) {
    return diff128(a.lo, b.lo) || diff128(a.mid, b.mid) || diff128(a.hi, b.hi);
}

static really_inline int isnonzero384(m384 a) {
    return isnonzero128(or128(or128(a.lo, a.mid), a.hi));
}

/**
 * "Rich" version of diff384(). Takes two vectors a and b and returns a 12-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich384(m384 a, m384 b) {
	u32 s0 = diffrich128(a.hi, b.hi);
	u32 s1 = diffrich128(a.mid, b.mid);
	u32 s2 = diffrich128(a.lo, b.lo);
    return (s0<<8)|(s1<<4)|s2;
}

/**
 * "Rich" version of diff384(), 64-bit variant. Takes two vectors a and b and
 * returns a 12-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_384(m384 a, m384 b) {
    u32 d = diffrich384(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m384 load384(const void *ptr) {
    assert(ISALIGNED_16(ptr));
    m384 rv = {load128(ptr), load128((const char *)ptr + 16),
               load128((const char *)ptr + 32)};
    return rv;
}

// aligned store
static really_inline void store384(void *ptr, m384 a) {
    assert(ISALIGNED_16(ptr));
    ptr = assume_aligned(ptr, 16);
    *(m384 *)ptr = a;
}

// unaligned load
static really_inline m384 loadu384(const void *ptr) {
    m384 rv = {loadu128(ptr), loadu128((const char *)ptr + 16),
               loadu128((const char *)ptr + 32)};
    return rv;
}

// packed unaligned store of first N bytes
static really_inline void storebytes384(void *ptr, m384 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline m384 loadbytes384(const void *ptr, unsigned int n) {
    m384 a = zeroes384();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

// switches on bit N in the given vector.
static really_inline void setbit384(m384 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else if (n < 256) {
        sub = &ptr->mid;
    } else {
        sub = &ptr->hi;
    }
    setbit128(sub, n % 128);
}

// switches off bit N in the given vector.
static really_inline void clearbit384(m384 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else if (n < 256) {
        sub = &ptr->mid;
    } else {
        sub = &ptr->hi;
    }
    clearbit128(sub, n % 128);
}

// tests bit N in the given vector.
static really_inline char testbit384(m384 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo;
    } else if (n < 256) {
        sub = val.mid;
    } else {
        sub = val.hi;
    }
    return testbit128(sub, n % 128);
}

/****
 **** 512-bit Primitives
 ****/

static really_inline m512 zeroes512(void) {
    m512 rv = {zeroes256(), zeroes256()};
    return rv;
}

static really_inline m512 ones512(void) {
    m512 rv = {ones256(), ones256()};
    return rv;
}

static really_inline m512 and512(m512 a, m512 b) {
    m512 rv;
    rv.lo = and256(a.lo, b.lo);
    rv.hi = and256(a.hi, b.hi);
    return rv;
}

static really_inline m512 or512(m512 a, m512 b) {
    m512 rv;
    rv.lo = or256(a.lo, b.lo);
    rv.hi = or256(a.hi, b.hi);
    return rv;
}

static really_inline m512 xor512(m512 a, m512 b) {
    m512 rv;
    rv.lo = xor256(a.lo, b.lo);
    rv.hi = xor256(a.hi, b.hi);
    return rv;
}

static really_inline m512 not512(m512 a) {
    m512 rv;
    rv.lo = not256(a.lo);
    rv.hi = not256(a.hi);
    return rv;
}

static really_inline m512 andnot512(m512 a, m512 b) {
    m512 rv;
    rv.lo = andnot256(a.lo, b.lo);
    rv.hi = andnot256(a.hi, b.hi);
    return rv;
}

static really_really_inline m512 lshift64_m512(m512 a, unsigned b) {
    m512 rv;
    rv.lo = lshift64_m256(a.lo, b);
    rv.hi = lshift64_m256(a.hi, b);
    return rv;
}

static really_inline int diff512(m512 a, m512 b) {
    return diff256(a.lo, b.lo) || diff256(a.hi, b.hi);
}

static really_inline int isnonzero512(m512 a) {
    m128 x = or128(a.lo.lo, a.lo.hi);
    m128 y = or128(a.hi.lo, a.hi.hi);
    return isnonzero128(or128(x, y));
}

/**
 * "Rich" version of diff512(). Takes two vectors a and b and returns a 16-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich512(m512 a, m512 b) {
	u32 s0 = diffrich256(a.lo, b.lo);
	u32 s1 = diffrich256(a.hi, b.hi);
    return s0|(s1<<8);
}

/**
 * "Rich" version of diffrich(), 64-bit variant. Takes two vectors a and b and
 * returns a 16-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_512(m512 a, m512 b) {
    u32 d = diffrich512(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m512 load512(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    m512 rv = {load256(ptr), load256((const char *)ptr + 32)};
    return rv;
}

// aligned store
static really_inline void store512(void *ptr, m512 a) {
    assert(ISALIGNED_N(ptr, alignof(m512)));
    ptr = assume_aligned(ptr, 16);
    *(m512 *)ptr = a;
}

// unaligned load
static really_inline m512 loadu512(const void *ptr) {
    m512 rv = {loadu256(ptr), loadu256((const char *)ptr + 32)};
    return rv;
}

// packed unaligned store of first N bytes
static really_inline void storebytes512(void *ptr, m512 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline m512 loadbytes512(const void *ptr, unsigned int n) {
    m512 a = zeroes512();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

static really_inline m512 mask1bit512(unsigned int n) {
    assert(n < sizeof(m512) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu512(&simd_onebit_masks[mask_idx]);
}

// switches on bit N in the given vector.
static really_inline void setbit512(m512 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo.lo;
    } else if (n < 256) {
        sub = &ptr->lo.hi;
    } else if (n < 384) {
        sub = &ptr->hi.lo;
    } else {
        sub = &ptr->hi.hi;
    }
    setbit128(sub, n % 128);
}

// switches off bit N in the given vector.
static really_inline void clearbit512(m512 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo.lo;
    } else if (n < 256) {
        sub = &ptr->lo.hi;
    } else if (n < 384) {
        sub = &ptr->hi.lo;
    } else {
        sub = &ptr->hi.hi;
    }
    clearbit128(sub, n % 128);
}

// tests bit N in the given vector.
static really_inline char testbit512(m512 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo.lo;
    } else if (n < 256) {
        sub = val.lo.hi;
    } else if (n < 384) {
        sub = val.hi.lo;
    } else {
        sub = val.hi.hi;
    }
    return testbit128(sub, n % 128);
}
#pragma GCC diagnostic pop

#endif

