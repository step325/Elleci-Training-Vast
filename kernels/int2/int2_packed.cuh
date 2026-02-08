#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace bitpack {

// ============================================================
// INT2 PACKING: 4 weights per byte
// Encoding: 00 = -1, 01 = 0, 10 = +1, 11 = reserved
// ============================================================

__device__ __forceinline__ int8_t unpack_int2(const uint8_t* data, int idx) {
    int byte_idx = idx >> 2;          // idx / 4
    int bit_offset = (idx & 3) << 1;  // (idx % 4) * 2
    uint8_t packed = data[byte_idx];
    uint8_t bits = (packed >> bit_offset) & 0x03;
    // Decode: 00->-1, 01->0, 10->+1
    return (int8_t)bits - 1;
}

__device__ __forceinline__ void pack_int2(uint8_t* data, int idx, int8_t val) {
    int byte_idx = idx >> 2;
    int bit_offset = (idx & 3) << 1;
    uint8_t encoded = (uint8_t)(val + 1);  // -1->0, 0->1, +1->2

    // Atomic RMW to avoid race conditions on same byte
    uint8_t mask = ~(0x03 << bit_offset);
    atomicAnd((unsigned int*)(data + (byte_idx & ~3)),
              ~(0x03u << (bit_offset + 8*(byte_idx & 3))));
    atomicOr((unsigned int*)(data + (byte_idx & ~3)),
             (unsigned int)encoded << (bit_offset + 8*(byte_idx & 3)));
}

// Non-atomic version (use when no race conditions)
__device__ __forceinline__ void pack_int2_unsafe(uint8_t* data, int idx, int8_t val) {
    int byte_idx = idx >> 2;
    int bit_offset = (idx & 3) << 1;
    uint8_t encoded = (uint8_t)(val + 1);
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
    uint8_t bits = (uint8_t)(val & 0x0F);
    if (nibble) {
        data[byte_idx] = (data[byte_idx] & 0x0F) | (bits << 4);
    } else {
        data[byte_idx] = (data[byte_idx] & 0xF0) | bits;
    }
}

// ============================================================
// BATCH UNPACK: Per matmul tiles
// ============================================================

// Unpack 4 weights from 1 byte into 4 int8 values
__device__ __forceinline__ void unpack_int2_x4(uint8_t packed, int8_t out[4]) {
    out[0] = (int8_t)((packed >> 0) & 0x03) - 1;
    out[1] = (int8_t)((packed >> 2) & 0x03) - 1;
    out[2] = (int8_t)((packed >> 4) & 0x03) - 1;
    out[3] = (int8_t)((packed >> 6) & 0x03) - 1;
}

// Unpack 4 weights directly to float for compute
__device__ __forceinline__ void unpack_int2_x4_float(uint8_t packed, float out[4]) {
    out[0] = (float)((int8_t)((packed >> 0) & 0x03) - 1);
    out[1] = (float)((int8_t)((packed >> 2) & 0x03) - 1);
    out[2] = (float)((int8_t)((packed >> 4) & 0x03) - 1);
    out[3] = (float)((int8_t)((packed >> 6) & 0x03) - 1);
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
