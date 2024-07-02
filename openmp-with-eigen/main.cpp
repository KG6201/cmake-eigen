#include <iostream>
#include <omp.h>
#include <Eigen/Core>

int main() {
    std::cout << "Control the number of threads" << std::endl;
    {
        // Default number of threads
        // Eigen 3.4.90 (latest) returns -1
        // Eigen 3.4.0 returns 8(same as omp_get_max_threads()?)
        std::cout << "Threads num(Eigen::nbThreads()): " << Eigen::nbThreads() << std::endl;

        std::cout << "Threads num(omp_get_num_threads()): " << omp_get_num_threads() << std::endl;
        std::cout << "Threads num(omp_get_max_threads()): " << omp_get_max_threads() << std::endl;
        std::cout << "call omp_set_num_threads(4)" << std::endl;
        omp_set_num_threads(4); // 4スレッドで計算
        std::cout << "Threads num(omp_get_num_threads()): " << omp_get_num_threads() << std::endl;
        std::cout << "Threads num(omp_get_max_threads()): " << omp_get_max_threads() << std::endl;
    }

    {
        Eigen::setNbThreads(1); // シングルスレッドで計算
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
    }

    {
        Eigen::setNbThreads(omp_get_num_threads()); // OpenMPのスレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
    }


    {
        Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
    }

    {
        omp_set_num_threads(2); // 2スレッドで計算
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
    }

    {
        Eigen::setNbThreads(omp_get_num_threads()); // OpenMPのスレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
    }

    {
        Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
    }

    {
        Eigen::setNbThreads(1); // シングルスレッドで計算
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
    }

    return 0;
}
