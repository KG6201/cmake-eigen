#include <iostream>
#include <omp.h>
#include <Eigen/Core>

int main() {
    std::cout << "Matrix multiplication" << std::endl;
    {
        constexpr int size = 1000;
        std::cout << "Matrix size: " << size << "x" << size << std::endl;
        Eigen::MatrixXd matrix1 = Eigen::MatrixXd::Random(size, size);
        Eigen::MatrixXd matrix2 = Eigen::MatrixXd::Random(size, size);
        Eigen::MatrixXd result(size, size);

        double start_time, end_time;

        // Eigenの内部並列化を使用
        Eigen::setNbThreads(1); // シングルスレッドで計算
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
        start_time = omp_get_wtime();
        result = matrix1 * matrix2;
        end_time = omp_get_wtime();
        std::cout << "Without parallelization (Eigen single thread): " << end_time - start_time << " seconds" << std::endl;

        Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
        start_time = omp_get_wtime();
        result = matrix1 * matrix2;
        end_time = omp_get_wtime();
        std::cout << "With parallelization (Eigen multi-thread): " << end_time - start_time << " seconds" << std::endl;
    }

    std::cout << "Vector inner product" << std::endl;
    {
        constexpr int size = 100000000;
        std::cout << "Vector size: " << size << std::endl;
        Eigen::VectorXd vector1 = Eigen::VectorXd::Random(size);
        Eigen::VectorXd vector2 = Eigen::VectorXd::Random(size);
        double result1, result2;

        double start_time, end_time;

        // Eigenの内部並列化を使用
        Eigen::setNbThreads(1); // シングルスレッドで計算
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
        start_time = omp_get_wtime();
        result1 = vector1.dot(vector2);
        end_time = omp_get_wtime();
        std::cout << "Without parallelization (Eigen single thread): " << end_time - start_time << " seconds" << std::endl;

        Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
        start_time = omp_get_wtime();
        result2 = vector1.dot(vector2);
        end_time = omp_get_wtime();
        std::cout << "With parallelization (Eigen multi-thread): " << end_time - start_time << " seconds" << std::endl;
        std::cout << "result1: " << result1 << ", result2: " << result2 << std::endl;
    }

    return 0;
}
