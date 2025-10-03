#include </home/mlir/MLIR/mlir-emitc/third_party/googletest/googletest/include/gtest/gtest.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>

// Scalar 4x4 transpose
void transpose4x4_scalar(const std::vector<float>& src, std::vector<float>& dst, int rows, int cols) {
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            dst[j*cols + i] = src[i*cols + j];
}

// AVX 4x4 transpose
void transpose4x4_avx(const std::vector<float>& src, std::vector<float>& dst, int rows, int cols) {
    __m128 row0 = _mm_loadu_ps(&src[0]);
    __m128 row1 = _mm_loadu_ps(&src[4]);
    __m128 row2 = _mm_loadu_ps(&src[8]);
    __m128 row3 = _mm_loadu_ps(&src[12]);

    __m128 tmp0 = _mm_unpacklo_ps(row0, row1);
    __m128 tmp1 = _mm_unpackhi_ps(row0, row1);
    __m128 tmp2 = _mm_unpacklo_ps(row2, row3);
    __m128 tmp3 = _mm_unpackhi_ps(row2, row3);

    row0 = _mm_movelh_ps(tmp0, tmp2);
    row1 = _mm_movehl_ps(tmp2, tmp0);
    row2 = _mm_movelh_ps(tmp1, tmp3);
    row3 = _mm_movehl_ps(tmp3, tmp1);

    _mm_storeu_ps(&dst[0], row0);
    _mm_storeu_ps(&dst[4], row1);
    _mm_storeu_ps(&dst[8], row2);
    _mm_storeu_ps(&dst[12], row3);
}

int main() {
    const int rows = 4, cols = 4;
    const int elements = rows * cols;
    const int iterations = 1000;

    std::vector<float> mat(elements), trans_scalar(elements), trans_avx(elements);

    std::cout << "Enter 4x4 matrix elements row-wise:\n";
    for(int i = 0; i < elements; ++i)
        std::cin >> mat[i];

    // Measure scalar transpose
    long long total_scalar_time = 0;
    for(int it = 0; it < iterations; ++it) {
        auto start = std::chrono::high_resolution_clock::now();
        transpose4x4_scalar(mat, trans_scalar, rows, cols);
        auto end = std::chrono::high_resolution_clock::now();
        total_scalar_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    double avg_scalar_time_ns = total_scalar_time / static_cast<double>(iterations);

    // Measure AVX transpose
    long long total_avx_time = 0;
    for(int it = 0; it < iterations; ++it) {
        auto start = std::chrono::high_resolution_clock::now();
        transpose4x4_avx(mat, trans_avx, rows, cols);
        auto end = std::chrono::high_resolution_clock::now();
        total_avx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    double avg_avx_time_ns = total_avx_time / static_cast<double>(iterations);

    // Print AVX transposed matrix
    std::cout << "\nTransposed 4x4 matrix (AVX last iteration):\n";
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j)
            std::cout << trans_avx[i*cols + j] << " ";
        std::cout << "\n";
    }

    std::cout << "Average scalar transpose: " << avg_scalar_time_ns << " ns\n";
    std::cout << "Average AVX transpose:    " << avg_avx_time_ns << " ns\n";
    std::cout << "Speedup: " << avg_scalar_time_ns / avg_avx_time_ns << "x\n";

    return 0;
}
