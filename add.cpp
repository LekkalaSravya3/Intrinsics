#include </home/mlir/MLIR/mlir-emitc/third_party/googletest/googletest/include/gtest/gtest.h>
#include <immintrin.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>

// addition
void add_matrices_scalar(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int elements) {
    for (int i = 0; i < elements; ++i) C[i] = A[i] + B[i];
}

// AVX addition
void add_matrices_avx(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int elements) {
    int simd_width = 8;
    int i = 0;
    for (; i <= elements - simd_width; i += simd_width) {
        __m256 vecA = _mm256_loadu_ps(&A[i]);
        __m256 vecB = _mm256_loadu_ps(&B[i]);
        __m256 vecC = _mm256_add_ps(vecA, vecB);
        _mm256_storeu_ps(&C[i], vecC);
    }
   
}

TEST(MatrixAddTest, AVXvsScalarIterations) {;
    const int elements = 1000;
    const int iterations = 20; 

    std::vector<float> A(elements), B(elements), C_scalar(elements), C_avx(elements);

    // Initialize random matrices
    std::srand(std::time(0));
    for (int i = 0; i < elements; ++i) {
        A[i] = static_cast<float>(std::rand()) / RAND_MAX * 100.0f;
        B[i] = static_cast<float>(std::rand()) / RAND_MAX * 100.0f;
    }

    // Measure scalar time
    long long total_scalar_time = 0;
    long long total_avx_time = 0;
    for (int it = 0; it < iterations; ++it) {
        auto start = std::chrono::high_resolution_clock::now();
        add_matrices_scalar(A, B, C_scalar, elements);
        auto end = std::chrono::high_resolution_clock::now();
        total_scalar_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        auto start_avx = std::chrono::high_resolution_clock::now();
        add_matrices_avx(A, B, C_avx, elements);
        auto end_avx = std::chrono::high_resolution_clock::now();
        total_avx_time += std::chrono::duration_cast<std::chrono::microseconds>(end_avx - start_avx).count();
    }
    double avg_scalar_time = total_scalar_time / static_cast<double>(iterations);
    double avg_avx_time = total_avx_time / static_cast<double>(iterations);

    std::cout << "Average scalar addition time : " << avg_scalar_time << " us\n";
    std::cout << "Average AVX addition time : " << avg_avx_time << " us\n";
    std::cout << "Speedup: " << avg_scalar_time / avg_avx_time << "x\n";

    for (int i = 0; i < elements; ++i) {
        EXPECT_EQ(C_scalar[i], C_avx[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
