// entropy_sha256.cl

// Helper function for rotating bits to the left
inline ulong rotate_left(ulong value, uint shift) {
    return (value << shift) | (value >> (64 - shift));
}

__kernel void gather_entropy(__global ulong *restrict entropy, __global ulong *restrict mem_block, __global ulong *restrict time_seed, __global uchar *restrict pixel_data, uint pixel_data_size) {
    uint gid = get_global_id(0);
    uint mem_size = get_global_size(0);

    // Initialize local entropy with a time-based seed for better randomness
    ulong local_entropy = time_seed[gid % get_global_size(1)] ^ (ulong)get_global_id(1) ^ (ulong)get_local_id(0);

    // Add entropy from pixel data already present in the GPU
    uint pixel_idx = gid % pixel_data_size;
    local_entropy ^= (ulong)pixel_data[pixel_idx] << 56;  // Use upper bits for higher impact

    // Precompute mask values for index calculation
    uint local_entropy_masked_7 = local_entropy & 0x7F; // 7 bits for shifting
    uint local_entropy_masked_13 = local_entropy & 0xFF; // 8 bits for index calc

    // Loop with unrolling for optimized performance
    #pragma unroll 16
    for (int i = 0; i < 1000; i++) {
        // Add perturbation based on pixel data
        local_entropy ^= rotate_left((ulong)pixel_data[(pixel_idx + i) % pixel_data_size], i % 64);

        // Calculate an unpredictable index
        uint idx = (gid + i + (local_entropy_masked_13 ^ (uint)((local_entropy >> 48) & 0xFF))) % mem_size;

        // Gather entropy using XOR and complex bitwise operations
        local_entropy ^= mem_block[idx];
        local_entropy ^= rotate_left(~local_entropy, (local_entropy_masked_7 + i) & 0x3F); // Variable shift mixing

        // Further diffusion with rotate and shift operations
        local_entropy = rotate_left(local_entropy, (local_entropy_masked_7 + 1) ^ (uint)((local_entropy >> 32) & 0x3F));
        local_entropy ^= (local_entropy << 13) ^ (local_entropy >> 7);

        // Introduce additional non-linearity
        local_entropy ^= (local_entropy * 31) + 0x9e3779b97f4a7c15; // A variant of the golden ratio

        // Update precomputed mask values
        local_entropy_masked_7 = local_entropy & 0x7F;
        local_entropy_masked_13 = local_entropy & 0xFF;
    }

    // Final perturbation before writing out
    local_entropy ^= rotate_left((ulong)pixel_data[pixel_idx], local_entropy_masked_7);

    // Write the gathered entropy to the output buffer
    entropy[gid] = local_entropy;
}

__constant uint k[64] = {
    // Constants for SHA-256
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__kernel void sha256(__global const uchar *input, __global uchar *output, uint num_blocks) {
    uint w[64];
    uint h[8] = {
        // Initial hash values for SHA-256
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint a, b, c, d, e, f, g, h0, t1, t2;

    for (uint block = 0; block < num_blocks; block++) {
        // Load input into the first 16 words
        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            w[i] = (input[block * 64 + 4*i] << 24) | (input[block * 64 + 4*i + 1] << 16) |
                   (input[block * 64 + 4*i + 2] << 8) | input[block * 64 + 4*i + 3];
        }

        // Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
        #pragma unroll 48
        for (int i = 16; i < 64; i++) {
            uint s0 = (w[i-15] >> 7 | w[i-15] << (32 - 7)) ^ (w[i-15] >> 18 | w[i-15] << (32 - 18)) ^ (w[i-15] >> 3);
            uint s1 = (w[i-2] >> 17 | w[i-2] << (32 - 17)) ^ (w[i-2] >> 19 | w[i-2] << (32 - 19)) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }

        a = h[0];
        b = h[1];
        c = h[2];
        d = h[3];
        e = h[4];
        f = h[5];
        g = h[6];
        h0 = h[7];

        #pragma unroll 64
        for (int i = 0; i < 64; i++) {
            t1 = h0 + ((e >> 6 | e << (32 - 6)) ^ (e >> 11 | e << (32 - 11)) ^ (e >> 25 | e << (32 - 25))) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            t2 = ((a >> 2 | a << (32 - 2)) ^ (a >> 13 | a << (32 - 13)) ^ (a >> 22 | a << (32 - 22))) + ((a & b) ^ (a & c) ^ (b & c));
            h0 = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h0;
    }

    // Output the final hash
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        output[4*i]     = (h[i] >> 24) & 0xff;
        output[4*i + 1] = (h[i] >> 16) & 0xff;
        output[4*i + 2] = (h[i] >> 8) & 0xff;
        output[4*i + 3] = h[i] & 0xff;
    }
}
