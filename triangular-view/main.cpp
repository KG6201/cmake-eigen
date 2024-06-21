#include <iostream>
#include <Eigen/Core>

int main() {
    // Create a 5x5 matrix of type double
    Eigen::MatrixXd matrix(5, 5);
    // Set values for the matrix
    matrix << 0, 2, 3, 4, 5,
              1, 0, 3, 4, 5,
              1, 2, 0, 4, 5,
              1, 2, 3, 0, 5,
              1, 2, 3, 4, 0;

    // Print the matrix
    std::cout << "Original Matrix:\n" << matrix << std::endl;

    // Create a skew-symmetric matrix from the original matrix
    matrix.triangularView<Eigen::StrictlyLower>() = - matrix.transpose();
    std::cout << "Skew-symmetric Matrix:\n" << matrix << std::endl;

    // Create a strictly-upper matrix from the skew-symmetric matrix
    matrix.triangularView<Eigen::StrictlyLower>().setZero();
    std::cout << "Strictly-upper Matrix:\n" << matrix << std::endl;

    return 0;
}
