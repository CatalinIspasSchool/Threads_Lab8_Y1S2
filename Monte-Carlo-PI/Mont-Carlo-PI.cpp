//CMP202, Mont Carlo Simulation for PI Estimation
// j.zarrin@abertay.ac.uk
//See https://en.wikipedia.org/wiki/Pi#/media/File:Pi_30K.gif 
// PI = 3.1415926535897932384626433...
// The program uses Monte Carlo simulation to estimate PI by calculating the ratio of points that fall inside a quarter circle to 
// the total number of points generated. The ratio is then used to estimate PI.
#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <chrono>


using namespace sycl;

int main() {
    // Define the number of points to generate. Can vary to test performance.
    // Experiment with different sizes  1e8, 1e7, 1e6, 1e5, 1e4, 1e3, and measure execution time for both CPU and GPU
    const size_t num_points = 1e1; // 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2
    
    // Vector to store the count of points that fall inside the circle.
    std::vector<size_t> num_points_in_circle_vec(1, 0);

    // Create a SYCL queue to submit work to. Here, we use a GPU selector,
    // Experiment changing this to cpu_selector{} to run on a CPU.
    queue q(gpu_selector{}, [](const exception_list& eL) {
        for (auto& e : eL) {
            try {
                std::rethrow_exception(e);
            }
            catch (const std::exception& e) {
                std::cerr << "SYCL exception: " << e.what() << std::endl;
            }
        }
        });

    // Print the device where the code is running.
    std::cout << "Running on " << q.get_device().get_info<info::device::name>() << std::endl;

    // Record the start time to measure execution time.
    auto start = std::chrono::high_resolution_clock::now();

    // Here is the main computation block. We declare a buffer within a scope so 
    // that when it goes out of scope, the buffer will be destructed, 
    // and its content will be automatically synchronized with the host memory data.
    { // Start of the buffer scope
        // Create a buffer for counting points inside the circle.
       
        //todo-1: Define the buffer here -------------------------------------------------------------
        // Submit a command group to the queue to execute the kernel.
        event e = q.submit([&](handler& h) {
            // Get access to the buffer.
            
            //todo-2: Get access to the buffer by defining an accessor-----------------------------------------

            // Launch a basic kernel (we use range object) parallel computation where each work-item represents a point.
            h.parallel_for(range<1>(num_points), [=](id<1> idx) {
                // Initialize a random number generator for each point.
                std::mt19937 rng(idx[0]);
                std::uniform_real_distribution<float> dist(0.0, 1.0);

                // Generate a random point within the unit square.
                float x = dist(rng);
                float y = dist(rng);

                // Check if the point is inside the quarter circle.
                if (x * x + y * y <= 1.0) {
                    // Safely increment the count using an atomic operation.

                    //todo-3 ------------------------------------------------------------
                    //acc[0]++;
                    
                }
                });
            });

        // Wait for the computation to complete.
        e.wait();
    }

    // Measure the end time and compute the execution time.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execution_time = end - start;

    // Calculate the estimate of Pi using the ratio of points inside the circle to the total number of points.
    long double pi_estimate = 4.0 * num_points_in_circle_vec[0] / num_points;


    std::cout << "Estimated Pi = " << pi_estimate << std::endl;
    std::cout << "Execution time: " << execution_time.count() << " ms" << std::endl;
    std::cout << "Number of points calculated: " << num_points << std::endl;

    return 0;
}
