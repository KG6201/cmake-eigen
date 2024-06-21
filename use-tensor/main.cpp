#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
    {
        Eigen::Tensor<int, 2> a(4, 3);
        a.setValues({{0, 100, 200}, {300, 400, 500},
                    {600, 700, 800}, {900, 1000, 1100}});
        Eigen::array<Eigen::Index, 2> offsets = {1, 0};
        Eigen::array<Eigen::Index, 2> extents = {2, 2};
        Eigen::Tensor<int, 2> slice = a.slice(offsets, extents);
        std::cout << "a" << std::endl << a << std::endl;
        std::cout << "slice([1:3), [0:2))" << std::endl << slice << std::endl;
    }

    {
        Eigen::Tensor<double, 2> tensor2d(2, 3);
        tensor2d.setValues(
            {
                {1, 2, 3},
                {4, 5, 6}
            }
        );
        std::cout << "tensor2d" << std::endl << tensor2d << std::endl;
        Eigen::array<Eigen::Index, 2> offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> extents = {2, 1};
        Eigen::Tensor<double, 2> slice = tensor2d.slice(offsets, extents);
        std::cout << "slice([0:2), [0:1))" << std::endl << slice << std::endl;
    }

    {
        Eigen::Tensor<double, 3> tensor3d(2, 3, 2);
        tensor3d.setValues(
            {
                {
                    {1, 2},
                    {3, 4},
                    {5, 6}
                },
                {
                    {7, 8},
                    {9, 10},
                    {11, 12}
                }
            }
        );
        Eigen::array<Eigen::Index, 3> offsets = {1, 1, 0};
        Eigen::array<Eigen::Index, 3> extents = {1, 1, 2};
        Eigen::Tensor<double, 3> slice = tensor3d.slice(offsets, extents);
        std::cout << "slice([1:2), [1:2), [0:2))" << std::endl << slice << std::endl;
    }

    {
        Eigen::Tensor<double, 4> tensor4d(2, 3, 2, 3);
        tensor4d.setValues(
            {{{
                {1, 2, 3},
                {4, 5, 6}
            }, {
                {7, 8, 9},
                {10, 11, 12}
            }, {
                {13, 14, 15},
                {16, 17, 18}
            }},
            {{
                {19, 20, 21},
                {22, 23, 24}
            }, {
                {25, 26, 27},
                {28, 29, 30}
            }, {
                {31, 32, 33},
                {34, 35, 36}
            }}}
        );
        Eigen::array<Eigen::Index, 4> offsets = {1, 0, 1, 0};
        Eigen::array<Eigen::Index, 4> extents = {1, 1, 1, 3};
        Eigen::Tensor<double, 4> slice = tensor4d.slice(offsets, extents);
        std::cout << "slice([1:2), [0:1), [1:2), [0:3))" << std::endl << slice << std::endl;
        std::cout << "sum: " << slice.sum() << std::endl;
    }

    {
        Eigen::Tensor<double, 4> tensor4d(2, 3, 2, 3);
        tensor4d.setValues(
            {{{
                {1, 2, 3},
                {4, 5, 6}
            }, {
                {7, 8, 9},
                {10, 11, 12}
            }, {
                {13, 14, 15},
                {16, 17, 18}
            }},
            {{
                {19, 20, 21},
                {22, 23, 24}
            }, {
                {25, 26, 27},
                {28, 29, 30}
            }, {
                {31, 32, 33},
                {34, 35, 36}
            }}}
        );
        for (size_t i = 0; i < static_cast<size_t>(tensor4d.dimension(0)); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(tensor4d.dimension(1)); ++j) {
                for (size_t k = 0; k < static_cast<size_t>(tensor4d.dimension(2)); ++k) {
                    Eigen::array<Eigen::Index, 4> offsets = {static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k), 0};
                    Eigen::array<Eigen::Index, 4> extents = {1, 1, 1, 3};
                    Eigen::Tensor<double, 4> slice = tensor4d.slice(offsets, extents);
                    std::cout << "sum(tensor(" << i << ", " << j << ", " << k << ", i)):\n" << slice.sum() << std::endl;
                }
            }
        }

        Eigen::array<Eigen::Index, 4> offsets = {1, 0, 1, 1};
        Eigen::array<Eigen::Index, 4> extents = {1, 1, 1, 2};
        Eigen::Tensor<double, 4> slice = tensor4d.slice(offsets, extents);
        std::cout << "slice([1:2), [0:1), [1:2), [1:3))" << std::endl << slice << std::endl;

        // Print the matrix
        std::cout << "4 rank Tensor:\n" << tensor4d << std::endl;
        for (size_t i = 0; i < static_cast<size_t>(tensor4d.dimension(0)); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(tensor4d.dimension(1)); ++j) {
                for (size_t k = 0; k < static_cast<size_t>(tensor4d.dimension(2)); ++k) {
                    for (size_t l = 0; l < static_cast<size_t>(tensor4d.dimension(3)); ++l) {
                        std::cout << "tensor4d(" << i << ", " << j << ", " << k << ", " << l << "): " << tensor4d(i, j, k, l) << std::endl;
                    }
                }
            }
        }
    }

    return 0;
}
