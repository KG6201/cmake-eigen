#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/SparseCore>

int main() {
    const int small_n = 5;
    // Create a nxn matrix of type double
    Eigen::SparseMatrix<double> matrix(small_n, small_n);

    // Set values at specific indices
    std::vector< Eigen::Triplet<double>> tripletList;
    tripletList.reserve(small_n * small_n - small_n);
    tripletList.push_back(Eigen::Triplet<double>(0, 1, 2));
    tripletList.push_back(Eigen::Triplet<double>(0, 4, 5));
    tripletList.push_back(Eigen::Triplet<double>(1, 2, 6));
    tripletList.push_back(Eigen::Triplet<double>(3, 4, 8));

    matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    // Print the matrix
    std::cout << "Original Matrix:\n" << matrix << std::endl;
    std::cout << "Transposed Matrix:\n" << matrix.transpose() << std::endl;

    // Edit specific indices
    matrix.coeffRef(0, 1) = 3;
    matrix.coeffRef(1, 2) = 7;
    matrix.insert(2, 3) = 9;

    // Print the edited matrix
    std::cout << "Edited Matrix:\n" << matrix << std::endl;

    // Sum up rowwise and colwise
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(small_n);
    Eigen::VectorXd rowSum = matrix.triangularView<Eigen::StrictlyUpper>() * ones;
    Eigen::VectorXd colSum = matrix.transpose().triangularView<Eigen::StrictlyLower>() * ones;

    // Print the rowwise and colwise sum
    std::cout << "Rowwise Sum:\n" << rowSum << std::endl;
    std::cout << "Colwise Sum:\n" << colSum << std::endl;

    // Reset the matrix and triplets to zero
    matrix.setZero();
    std::cout << "Triplets Size: " << tripletList.size() << ", Capacity: " << tripletList.capacity() << std::endl;
    tripletList.clear();
    std::cout << "Triplets Size after clear: " << tripletList.size() << ", Capacity: " << tripletList.capacity() << std::endl;

    // Set values at specific indices
    tripletList.push_back(Eigen::Triplet<double>(0, 1, 2));
    tripletList.push_back(Eigen::Triplet<double>(0, 4, 5));
    tripletList.push_back(Eigen::Triplet<double>(1, 2, 6));
    tripletList.push_back(Eigen::Triplet<double>(3, 4, 8));
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    // Print the matrix
    std::cout << "Original Matrix:\n" << matrix << std::endl;

    // Create a skew-symmetric matrix
    Eigen::SparseMatrix<double> skewSymmetric(small_n, small_n);
    skewSymmetric.setFromTriplets(tripletList.begin(), tripletList.end());
    skewSymmetric -= Eigen::SparseMatrix<double>(skewSymmetric.transpose()).triangularView<Eigen::StrictlyLower>();
    std::cout << "Skew-Symmetric Matrix:\n" << skewSymmetric << std::endl;

    // Sum up rowwise
    Eigen::VectorXd skewSymmetricRowSum = skewSymmetric * ones;

    // Print the rowwise sum
    std::cout << "Skew-Symmetric Rowwise Sum:\n" << skewSymmetricRowSum << std::endl;

    // Create large sparse matrix
    const int large_n = 10000;
    Eigen::SparseMatrix<double> largeSparseMatrix(large_n, large_n);
    // Set random values
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0,1.0);

    std::vector<Eigen::Triplet<double>> largeTripletList;
    for(size_t i = 0; i < large_n; ++i) {
        for(size_t j = 0; j < large_n; ++j) {
            auto v_ij = dist(gen);
            if(v_ij < 0.1) {
                largeTripletList.push_back(Eigen::Triplet<double>(i, j, v_ij));
            }
        }
    }
    largeSparseMatrix.setFromTriplets(largeTripletList.begin(), largeTripletList.end());   //create the matrix

    // Print number of non-zero elements
    std::cout << "Number of Non-Zero Elements: " << largeSparseMatrix.nonZeros() << std::endl;

    // Create large dense matrix
    Eigen::MatrixXd largeDenseMatrix = largeSparseMatrix;

    {
        // Measure time for matrix multiplication
        Eigen::VectorXd x = Eigen::VectorXd::Random(large_n);
        Eigen::VectorXd y;

        // Sparse Matrix Multiplication
        auto start = std::chrono::high_resolution_clock::now();
        y = largeSparseMatrix * x;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Sparse Matrix Multiplication Time: " << elapsed_seconds.count() << "s" << std::endl;

        // Dense Matrix Multiplication
        start = std::chrono::high_resolution_clock::now();
        y = largeDenseMatrix * x;
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Dense Matrix Multiplication Time: " << elapsed_seconds.count() << "s" << std::endl;
    }

    // Try skew-symmetric matrix
    // Make matrix strictly upper triangular
    largeSparseMatrix = largeSparseMatrix.triangularView<Eigen::StrictlyUpper>();
    largeDenseMatrix = largeSparseMatrix;

    // Print number of non-zero elements
    std::cout << "Number of Non-Zero Elements after making it strictly upper triangular: " << largeSparseMatrix.nonZeros() << std::endl;

    {
        // Measure time for rowwise sum
        Eigen::VectorXd largeOnes = Eigen::VectorXd::Ones(large_n);
        Eigen::VectorXd rowSumSparse, rowSumDense;

        // Sparse Matrix Multiplication
        auto start = std::chrono::high_resolution_clock::now();
        rowSumSparse = largeSparseMatrix * largeOnes - largeSparseMatrix.transpose() * largeOnes;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Sparse Matrix Rowwise Sum Time: " << elapsed_seconds.count() << "s" << std::endl;

        // Dense Matrix Multiplication
        start = std::chrono::high_resolution_clock::now();
        rowSumDense = largeDenseMatrix.rowwise().sum() - largeDenseMatrix.colwise().sum().transpose();
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Dense Matrix Rowwise Sum Time: " << elapsed_seconds.count() << "s" << std::endl;

        // Check if the rowwise sum is the same
        std::cout << "Rowwise Sum for Sparse and Dense Matrix are the same: " << static_cast<bool>((rowSumSparse - rowSumDense).norm() < 1e-6) << std::endl;
        std::cout << "Rowwise Sum for Sparse Matrix:\n" << rowSumSparse.norm() << std::endl;
        std::cout << "Rowwise Sum for Dense Matrix:\n" << rowSumDense.norm() << std::endl;

        // Sparse Matrix Multiplication
        start = std::chrono::high_resolution_clock::now();
        largeSparseMatrix -= Eigen::SparseMatrix<double>(largeSparseMatrix.transpose()).triangularView<Eigen::StrictlyLower>();
        rowSumSparse = largeSparseMatrix * largeOnes;
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Sparse Matrix Rowwise Sum Time Using Skew-Symmetric Matrix: " << elapsed_seconds.count() << "s" << std::endl;

        // Dense Matrix Multiplication
        start = std::chrono::high_resolution_clock::now();
        largeDenseMatrix -= largeDenseMatrix.transpose().triangularView<Eigen::StrictlyLower>();
        rowSumDense = largeDenseMatrix.rowwise().sum();
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Dense Matrix Rowwise Sum Time Using Skew-Symmetric Matrix: " << elapsed_seconds.count() << "s" << std::endl;

        // Check if the rowwise sum is the same
        std::cout << "Rowwise Sum for Sparse and Dense Matrix are the same: " << static_cast<bool>((rowSumSparse - rowSumDense).norm() < 1e-6) << std::endl;
        std::cout << "Rowwise Sum for Sparse Matrix:\n" << rowSumSparse.norm() << std::endl;
        std::cout << "Rowwise Sum for Dense Matrix:\n" << rowSumDense.norm() << std::endl;
    }

    return 0;
}
