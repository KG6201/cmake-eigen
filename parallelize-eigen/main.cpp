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
        Eigen::MatrixXd result_serial(size, size);
        Eigen::MatrixXd result_parallel(size, size);

        double start_time, end_time;

        // Eigenの内部並列化を使用
        {
            Eigen::setNbThreads(1); // シングルスレッドで計算
            std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
            start_time = omp_get_wtime();
            result_serial = matrix1 * matrix2;
            end_time = omp_get_wtime();
            std::cout << "Without parallelization (Eigen single thread): " << end_time - start_time << " seconds" << std::endl;
        }

        {
            Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
            std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
            start_time = omp_get_wtime();
            result_parallel = matrix1 * matrix2;
            end_time = omp_get_wtime();
            std::cout << "With parallelization (Eigen multi-thread): " << end_time - start_time << " seconds" << std::endl;
        }
        std::cout << "result_serial(0, 0): " << result_serial(0, 0) << ", result_parallel(0, 0): " << result_parallel(0, 0) << std::endl;
        std::cout << "sum of result_serial: " << result_serial.sum() << ", sum of result_parallel: " << result_parallel.sum() << std::endl;
    }

    std::cout << "Vector inner product" << std::endl;
    {
        constexpr int size = 100000000;
        std::cout << "Vector size: " << size << std::endl;
        Eigen::VectorXd vector1 = Eigen::VectorXd::Random(size);
        Eigen::VectorXd vector2 = Eigen::VectorXd::Random(size);
        double result_serial, result_parallel;

        double start_time, end_time;

        // Eigenの内部並列化を使用
        {
            Eigen::setNbThreads(1); // シングルスレッドで計算
            std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
            start_time = omp_get_wtime();
            result_serial = vector1.dot(vector2);
            end_time = omp_get_wtime();
            std::cout << "Without parallelization (Eigen single thread): " << end_time - start_time << " seconds" << std::endl;
        }

        {
            Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
            std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
            start_time = omp_get_wtime();
            result_parallel = vector1.dot(vector2);
            end_time = omp_get_wtime();
            std::cout << "With parallelization (Eigen multi-thread): " << end_time - start_time << " seconds" << std::endl;
        }
        std::cout << "result_serial: " << result_serial << ", result_parallel: " << result_parallel << std::endl;
    }

    return 0;
}
