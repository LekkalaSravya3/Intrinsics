#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
                          __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {

//  _mm256_unpacklo_ps(a, b) → interleaves lower 4 elements (lane-wise) of a and b.
// → Result: [a0, b0, a1, b1, a2, b2, a3, b3]

// _mm256_unpackhi_ps(a, b) → interleaves higher 4 elements of a and b.
// → Result: [a4, b4, a5, b5, a6, b6, a7, b7]

// These help pair up rows (row0 with row1, row2 with row3, etc.) — essentially forming 2×2 sub-blocks of the transpose.
    __m256 t0 = _mm256_unpacklo_ps(row0, row1);
    __m256 t1 = _mm256_unpackhi_ps(row0, row1);
    __m256 t2 = _mm256_unpacklo_ps(row2, row3);
    __m256 t3 = _mm256_unpackhi_ps(row2, row3);
    __m256 t4 = _mm256_unpacklo_ps(row4, row5);
    __m256 t5 = _mm256_unpackhi_ps(row4, row5);
    __m256 t6 = _mm256_unpacklo_ps(row6, row7);
    __m256 t7 = _mm256_unpackhi_ps(row6, row7);

    // _mm256_shuffle_ps(a, b, imm) → shuffles elements from a and b based on the immediate control value imm.
    // _MM_SHUFFLE(z, y, x, w) → creates an immediate value for shuffling.
    // This step rearranges the data to prepare for the final permutation.

    __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
    __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
    __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
    __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
    __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
    __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
    __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
    __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

    // _mm256_permute2f128_ps(a, b, imm) → rearranges 128-bit halves (lanes) between two 256-bit registers.
    // 0x20 → [b.low | a.low]
    // 0x31 → [b.high | a.high]
    // These combine data from the lower and upper 128-bit lanes of the shuffled results to form the final 8×8 transposed output.

    row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

void transpose_scalar(const float in[8][8], float out[8][8]) {
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            out[c][r] = in[r][c];
}

// Each row of 8 floats is loaded into a 256-bit AVX register (__m256), which holds 8 floats (8 × 32 bits = 256 bits).
// So, the entire 8×8 matrix fits into 8 AVX registers:
void transpose_avx_8x8(const float in[8][8], float out[8][8]) {
    // loads 8 floats (256 bits) from memory into an AVX register.
    __m256 r0 = _mm256_loadu_ps(in[0]);
    __m256 r1 = _mm256_loadu_ps(in[1]);
    __m256 r2 = _mm256_loadu_ps(in[2]);
    __m256 r3 = _mm256_loadu_ps(in[3]);
    __m256 r4 = _mm256_loadu_ps(in[4]);
    __m256 r5 = _mm256_loadu_ps(in[5]);
    __m256 r6 = _mm256_loadu_ps(in[6]);
    __m256 r7 = _mm256_loadu_ps(in[7]);

    transpose8_ps(r0, r1, r2, r3, r4, r5, r6, r7);

    // stores 8 floats (256 bits) from an AVX register to memory.
    _mm256_storeu_ps(out[0], r0);
    _mm256_storeu_ps(out[1], r1);
    _mm256_storeu_ps(out[2], r2);
    _mm256_storeu_ps(out[3], r3);
    _mm256_storeu_ps(out[4], r4);
    _mm256_storeu_ps(out[5], r5);
    _mm256_storeu_ps(out[6], r6);
    _mm256_storeu_ps(out[7], r7);
}

TEST(MatrixTransposeTest, AVX8x8_vs_Scalar) {
    float A[8][8];
    float B_scalar[8][8];
    float B_avx[8][8];

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            A[i][j] = static_cast<float>(std::rand()) / RAND_MAX;

    const int iterations = 100;
    long long scalar_time = 0;
    long long avx_time = 0;

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        transpose_scalar(A, B_scalar);
        auto t1 = std::chrono::high_resolution_clock::now();
        scalar_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        auto t0_avx = std::chrono::high_resolution_clock::now();
        transpose_avx_8x8(A, B_avx);
        auto t1_avx = std::chrono::high_resolution_clock::now();
        avx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1_avx - t0_avx).count();
    }

    double avg_scalar = scalar_time / static_cast<double>(iterations);
    double avg_avx = avx_time / static_cast<double>(iterations);
    double speedup = avg_scalar / avg_avx;

    std::cout << "\nAverage Scalar Transpose Time: " << avg_scalar << " ns\n";
    std::cout << "Average AVX Transpose Time:    " << avg_avx << " ns\n";
    std::cout << "Speedup: " << speedup << "x\n";
// This Approach is 6-7x faster than the scalar version.
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            EXPECT_NEAR(B_scalar[i][j], B_avx[i][j], 1e-5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
