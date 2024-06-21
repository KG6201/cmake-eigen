#include <iostream>
#include <Eigen/Core>

int main() {
    // Create a 3x3 matrix of type double
    Eigen::Matrix3d matrix;

    // Set values for the matrix
    matrix << 0, 2, 3,
              0, 0, 6,
              0, 0, 0;

    // Print the matrix
    std::cout << "Original Matrix:\n" << matrix << std::endl;
    std::cout << "Transposed Matrix:\n" << matrix.transpose() << std::endl;

    // Create a skew-symmetric matrix from the original matrix
    matrix -= matrix.transpose().eval();
    std::cout << "Skew-symmetric Matrix:\n" << matrix << std::endl;

    return 0;
}
