#include <iostream>
#include <vector>
#include <utility>
#include <numeric>
#include <omp.h>
#include <Eigen/Core>

int main() {
    std::cout << "Matrix multiplication" << std::endl;
    {
        constexpr int num_of_rods = 10000;
        constexpr int num_of_segments = 10;
        constexpr int size = num_of_rods * num_of_segments;
        std::cout << "Vector size: " << size << std::endl;
        srand((unsigned int) time(0));
        Eigen::VectorXd rods_segments = Eigen::VectorXd::Random(size);
        std::vector<double> displacement_rods_segments_std_vector_serial(size, 0.0);
        std::vector<double> displacement_rods_segments_std_vector_parallel(size, 0.0);
        Eigen::VectorXd displacement_rods_segments_no_pragma = Eigen::VectorXd::Zero(size);
        Eigen::VectorXd displacement_rods_segments_parallel = Eigen::VectorXd::Zero(size);
        Eigen::VectorXd displacement_rods_segments_via_std_vector = Eigen::VectorXd::Zero(size);

        std::vector<std::pair<size_t, size_t>> segment_partition_indices(num_of_rods);
        for (size_t i = 0; i < segment_partition_indices.size(); i++) {
            segment_partition_indices.at(i) = std::make_pair(i * num_of_segments, (i + 1) * num_of_segments);
        }

        double start_time, end_time;

        // Without pragma omp parallel for using std::vector
        start_time = omp_get_wtime();
        for (size_t rod_i = 0; rod_i < segment_partition_indices.size(); rod_i++) {
            for (size_t rod_k = rod_i + 1; rod_k < segment_partition_indices.size(); rod_k++) {
                for (size_t segment_j = segment_partition_indices.at(rod_i).first; segment_j < segment_partition_indices.at(rod_i).second; segment_j++) {
                    for (size_t segment_l = segment_partition_indices.at(rod_k).first; segment_l < segment_partition_indices.at(rod_k).second; segment_l++) {
                        // calculate the distance between segment j in rod i and segment l in rod k
                        const double x1 = rods_segments(segment_j);
                        const double x2 = rods_segments(segment_l);

                        const double dx = x2 - x1;
                        displacement_rods_segments_std_vector_serial.at(segment_j) += dx;
                        displacement_rods_segments_std_vector_serial.at(segment_l) -= dx;
                    }
                }
            }
        }
        end_time = omp_get_wtime();
        std::cout << "Without parallelization (std::vector single thread): " << end_time - start_time << " seconds" << std::endl;

        // With pragma omp parallel for using std::vector
        Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
        start_time = omp_get_wtime();
        #pragma omp declare reduction(+: std::vector<double>: std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), 0.0))
        #pragma omp parallel for reduction(+: displacement_rods_segments_std_vector_parallel) schedule(dynamic) num_threads(Eigen::nbThreads())
        for (size_t rod_i = 0; rod_i < segment_partition_indices.size(); rod_i++) {
            for (size_t rod_k = rod_i + 1; rod_k < segment_partition_indices.size(); rod_k++) {
                for (size_t segment_j = segment_partition_indices.at(rod_i).first; segment_j < segment_partition_indices.at(rod_i).second; segment_j++) {
                    for (size_t segment_l = segment_partition_indices.at(rod_k).first; segment_l < segment_partition_indices.at(rod_k).second; segment_l++) {
                        // calculate the distance between segment j in rod i and segment l in rod k
                        const double x1 = rods_segments(segment_j);
                        const double x2 = rods_segments(segment_l);

                        const double dx = x2 - x1;
                        displacement_rods_segments_std_vector_parallel.at(segment_j) += dx;
                        displacement_rods_segments_std_vector_parallel.at(segment_l) -= dx;
                    }
                }
            }
        }
        end_time = omp_get_wtime();
        std::cout << "With parallelization (std::vector multi-thread): " << end_time - start_time << " seconds" << std::endl;

        // Without pragma omp parallel for
        start_time = omp_get_wtime();
        for (size_t rod_i = 0; rod_i < segment_partition_indices.size(); rod_i++) {
            for (size_t rod_k = rod_i + 1; rod_k < segment_partition_indices.size(); rod_k++) {
                for (size_t segment_j = segment_partition_indices.at(rod_i).first; segment_j < segment_partition_indices.at(rod_i).second; segment_j++) {
                    for (size_t segment_l = segment_partition_indices.at(rod_k).first; segment_l < segment_partition_indices.at(rod_k).second; segment_l++) {
                        // calculate the distance between segment j in rod i and segment l in rod k
                        const double x1 = rods_segments(segment_j);
                        const double x2 = rods_segments(segment_l);

                        const double dx = x2 - x1;
                        displacement_rods_segments_no_pragma(segment_j) += dx;
                        displacement_rods_segments_no_pragma(segment_l) -= dx;
                    }
                }
            }
        }
        end_time = omp_get_wtime();
        std::cout << "Without pragma omp parallel for: " << end_time - start_time << " seconds" << std::endl;

        // With pragma omp parallel for using multi-thread (it does not work well due to without correct reduction)
        Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
        start_time = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic) num_threads(Eigen::nbThreads())
        for (size_t rod_i = 0; rod_i < segment_partition_indices.size(); rod_i++) {
            for (size_t rod_k = rod_i + 1; rod_k < segment_partition_indices.size(); rod_k++) {
                for (size_t segment_j = segment_partition_indices.at(rod_i).first; segment_j < segment_partition_indices.at(rod_i).second; segment_j++) {
                    for (size_t segment_l = segment_partition_indices.at(rod_k).first; segment_l < segment_partition_indices.at(rod_k).second; segment_l++) {
                        // calculate the distance between segment j in rod i and segment l in rod k
                        const double x1 = rods_segments(segment_j);
                        const double x2 = rods_segments(segment_l);

                        const double dx = x2 - x1;
                        displacement_rods_segments_parallel(segment_j) += dx;
                        displacement_rods_segments_parallel(segment_l) -= dx;
                    }
                }
            }
        }
        end_time = omp_get_wtime();
        std::cout << "With parallelization (Eigen multi-thread): " << end_time - start_time << " seconds" << std::endl;

        // With pragma omp parallel for using multi-thread and std::vector
        Eigen::setNbThreads(omp_get_max_threads()); // OpenMPの最大スレッド数を設定
        std::cout << "Threads num: " << Eigen::nbThreads() << std::endl;
        start_time = omp_get_wtime();
        {
            std::vector<double> displacement_rods_segments(size, 0.0);
            #pragma omp parallel for reduction(+: displacement_rods_segments) schedule(dynamic) num_threads(Eigen::nbThreads())
            for (size_t rod_i = 0; rod_i < segment_partition_indices.size(); rod_i++) {
                for (size_t rod_k = rod_i + 1; rod_k < segment_partition_indices.size(); rod_k++) {
                    for (size_t segment_j = segment_partition_indices.at(rod_i).first; segment_j < segment_partition_indices.at(rod_i).second; segment_j++) {
                        for (size_t segment_l = segment_partition_indices.at(rod_k).first; segment_l < segment_partition_indices.at(rod_k).second; segment_l++) {
                            // calculate the distance between segment j in rod i and segment l in rod k
                            const double x1 = rods_segments(segment_j);
                            const double x2 = rods_segments(segment_l);

                            const double dx = x2 - x1;
                            displacement_rods_segments.at(segment_j) += dx;
                            displacement_rods_segments.at(segment_l) -= dx;
                        }
                    }
                }
            }
            displacement_rods_segments_via_std_vector = Eigen::Map<Eigen::VectorXd>(displacement_rods_segments.data(), size);
        }
        end_time = omp_get_wtime();
        std::cout << "With parallelization (Eigen multi-thread and std::vector): " << end_time - start_time << " seconds" << std::endl;


        std::cout << "norm of displacement_rods_segments_std_vector_serial: " << std::sqrt(std::inner_product(displacement_rods_segments_std_vector_serial.begin(), displacement_rods_segments_std_vector_serial.end(), displacement_rods_segments_std_vector_serial.begin(), 0.0)) << std::endl;
        std::cout << "norm of displacement_rods_segments_std_vector_parallel: " << std::sqrt(std::inner_product(displacement_rods_segments_std_vector_parallel.begin(), displacement_rods_segments_std_vector_parallel.end(), displacement_rods_segments_std_vector_parallel.begin(), 0.0)) << std::endl;
        std::cout << "norm of displacement_rods_segments_no_pragma: " << displacement_rods_segments_no_pragma.norm() << std::endl;
        std::cout << "norm of displacement_rods_segments_parallel: " << displacement_rods_segments_parallel.norm() << std::endl;
        std::cout << "norm of displacement_rods_segments_via_std_vector: " << displacement_rods_segments_via_std_vector.norm() << std::endl;

        std::cout << "displacement_rods_segments_std_vector_serial(size / 2): " << displacement_rods_segments_std_vector_serial.at(size / 2) << std::endl;
        std::cout << "displacement_rods_segments_std_vector_parallel(size / 2): " << displacement_rods_segments_std_vector_parallel.at(size / 2) << std::endl;
        std::cout << "displacement_rods_segments_no_pragma(size / 2): " << displacement_rods_segments_no_pragma(size / 2) << std::endl;
        std::cout << "displacement_rods_segments_parallel(size / 2): " << displacement_rods_segments_parallel(size / 2) << std::endl;
        std::cout << "displacement_rods_segments_via_std_vector(size / 2): " << displacement_rods_segments_via_std_vector(size / 2) << std::endl;

        std::cout << "displacement_rods_segments_std_vector_serial(size-1): " << displacement_rods_segments_std_vector_serial.at(size-1) << std::endl;
        std::cout << "displacement_rods_segments_std_vector_parallel(size-1): " << displacement_rods_segments_std_vector_parallel.at(size-1) << std::endl;
        std::cout << "displacement_rods_segments_no_pragma(size-1): " << displacement_rods_segments_no_pragma(size-1) << std::endl;
        std::cout << "displacement_rods_segments_parallel(size-1): " << displacement_rods_segments_parallel(size-1) << std::endl;
        std::cout << "displacement_rods_segments_via_std_vector(size-1): " << displacement_rods_segments_via_std_vector(size-1) << std::endl;

        for (size_t i = 0; i < size; i++) {
            if (std::abs(displacement_rods_segments_std_vector_serial.at(i) - displacement_rods_segments_std_vector_parallel.at(i)) > 1.0e-7) {
                std::cout << "diff(std_vector_serial, std_vector_parallel) at " << i << ": " << displacement_rods_segments_std_vector_serial.at(i) - displacement_rods_segments_std_vector_parallel.at(i) << std::endl;
            }
            if (std::abs(displacement_rods_segments_std_vector_serial.at(i) - displacement_rods_segments_parallel(i)) > 1.0e-7) {
                // std::cout << "diff(std_vector_serial, parallel) at " << i << ": " << displacement_rods_segments_std_vector_serial.at(i) - displacement_rods_segments_parallel(i) << std::endl;
            }
            if (std::abs(displacement_rods_segments_std_vector_serial.at(i) - displacement_rods_segments_via_std_vector(i)) > 1.0e-7) {
                std::cout << "diff(std_vector_serial, via_std_vector) at " << i << ": " << displacement_rods_segments_std_vector_serial.at(i) - displacement_rods_segments_via_std_vector(i) << std::endl;
            }
        }
    }

    return 0;
}
