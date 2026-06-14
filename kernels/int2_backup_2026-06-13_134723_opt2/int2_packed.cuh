#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace bitpack {

// ============================================================
// INT2 PACKING: 4 weights per byte
// Encoding: 00 = -1, 01 = 0, 10 = +1, 11 = reserved (decoded as 0)
// ============================================================

__host__ __device__ __forceinline__ int8_t decode_int2_bits(uint8_t bits) {
    bits &= 0x03;
    if (bits == 0) return -1;
    if (bits == 2) return 1;
    return 0;
}

__device__ __forceinline__ int8_t unpack_int2(const uint8_t* data, int idx) {
    int byte_idx = idx >> 2;          // idx / 4
    int bit_offset = (idx & 3) << 1;  // (idx % 4) * 2
    uint8_t packed = data[byte_idx];
    uint8_t bits = (packed >> bit_offset) & 0x03;
    return decode_int2_bits(bits);
}

__device__ __forceinline__ void pack_int2(uint8_t* data, int idx, int8_t val) {
    int byte_idx = idx >> 2;
    int bit_offset = (idx & 3) << 1;
    int8_t clipped = (val < -1) ? -1 : ((val > 1) ? 1 : val);
    uint8_t encoded = (uint8_t)(clipped + 1);  // -1->0, 0->1, +1->2

    // Atomic RMW to avoid race conditions on same byte
    atomicAnd((unsigned int*)(data + (byte_idx & ~3)),
              ~(0x03u << (bit_offset + 8*(byte_idx & 3))));
    atomicOr((unsigned int*)(data + (byte_idx & ~3)),
             (unsigned int)encoded << (bit_offset + 8*(byte_idx & 3)));
}

// Non-atomic version (use when no race conditions)
__device__ __forceinline__ void pack_int2_unsafe(uint8_t* data, int idx, int8_t val) {
    int byte_idx = idx >> 2;
    int bit_offset = (idx & 3) << 1;
    int8_t clipped = (val < -1) ? -1 : ((val > 1) ? 1 : val);
    uint8_t encoded = (uint8_t)(clipped + 1);
    uint8_t mask = ~(0x03 << bit_offset);
    data[byte_idx] = (data[byte_idx] & mask) | (encoded << bit_offset);
}

// ============================================================
// INT4 PACKING: 2 hysteresis counters per byte
// Signed: [-8, +7]
// ============================================================

__device__ __forceinline__ int8_t unpack_int4(const uint8_t* data, int idx) {
    int byte_idx = idx >> 1;          // idx / 2
    int nibble = idx & 1;             // idx % 2
    uint8_t packed = data[byte_idx];
    uint8_t bits = nibble ? (packed >> 4) : (packed & 0x0F);
    // Sign extend from 4 bits
    return (bits & 0x08) ? (int8_t)(bits | 0xF0) : (int8_t)bits;
}

__device__ __forceinline__ void pack_int4_unsafe(uint8_t* data, int idx, int8_t val) {
    int byte_idx = idx >> 1;
    int nibble = idx & 1;
    int8_t clipped = (val < -8) ? -8 : ((val > 7) ? 7 : val);
    uint8_t bits = (uint8_t)(clipped & 0x0F);
    if (nibble) {
        data[byte_idx] = (data[byte_idx] & 0x0F) | (bits << 4);
    } else {
        data[byte_idx] = (data[byte_idx] & 0xF0) | bits;
    }
}

__device__ __forceinline__ void pack_int4(uint8_t* data, int idx, int8_t val) {
    int byte_idx = idx >> 1;
    int nibble = idx & 1;
    int8_t clipped = (val < -8) ? -8 : ((val > 7) ? 7 : val);
    uint8_t bits = (uint8_t)(clipped & 0x0F);
    int shift = (nibble ? 4 : 0) + 8 * (byte_idx & 3);

    atomicAnd((unsigned int*)(data + (byte_idx & ~3)), ~(0x0Fu << shift));
    atomicOr((unsigned int*)(data + (byte_idx & ~3)), (unsigned int)bits << shift);
}

// ============================================================
// BATCH UNPACK: Per matmul tiles
// ============================================================

// Unpack 4 weights from 1 byte into 4 int8 values
__device__ __forceinline__ void unpack_int2_x4(uint8_t packed, int8_t out[4]) {
    out[0] = decode_int2_bits((packed >> 0) & 0x03);
    out[1] = decode_int2_bits((packed >> 2) & 0x03);
    out[2] = decode_int2_bits((packed >> 4) & 0x03);
    out[3] = decode_int2_bits((packed >> 6) & 0x03);
}

// Unpack 4 weights directly to float for compute
__device__ __forceinline__ void unpack_int2_x4_float(uint8_t packed, float out[4]) {
    out[0] = (float)decode_int2_bits((packed >> 0) & 0x03);
    out[1] = (float)decode_int2_bits((packed >> 2) & 0x03);
    out[2] = (float)decode_int2_bits((packed >> 4) & 0x03);
    out[3] = (float)decode_int2_bits((packed >> 6) & 0x03);
}

// ============================================================
// MEMORY CALCULATIONS
// ============================================================

__host__ __device__ __forceinline__ size_t int2_packed_size(size_t num_weights) {
    return (num_weights + 3) / 4;  // Ceil division
}

__host__ __device__ __forceinline__ size_t int4_packed_size(size_t num_weights) {
    return (num_weights + 1) / 2;  // Ceil division
}

}  // namespace bitpack
