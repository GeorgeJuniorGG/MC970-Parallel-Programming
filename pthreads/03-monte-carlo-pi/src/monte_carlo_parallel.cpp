#include <cassert>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

std::mutex m;

typedef struct
{
  int *count;
  int *n_points;
  int num_iterations;
} WorkerArgs;

// Function to generate random numbers between -1 and 1
double random_number()
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return dis(gen);
}

// Function to estimate pi using the Monte Carlo method
void calculate_pi(WorkerArgs *const args)
{
  int hits = 0;
  int local_count = 0;
  int local_n_points = 0;

  for (int i = 0; i < args->num_iterations; ++i)
  {
    local_n_points++; // count every try

    double x = random_number();
    double y = random_number();

    if (x * x + y * y <= 1.0)
    {
      ++hits;
    }
  }

  local_count += hits;

  std::cout << "hits: " << hits << " of " << args->num_iterations << std::endl;

  int *count = args->count;
  int *n_points = args->n_points;

  // Add mutex to ensure no syncronization conflicts
  m.lock();
  *(count) += local_count;
  *(n_points) += local_n_points;
  m.unlock();
}

int main()
{
  const int num_iterations = 30000000;

  int count = 0;
  int n_points = 0;
  int num_threads = 4000; // set to get same result as parallel version

  std::thread workers[num_threads];
  WorkerArgs args[num_threads];

  for (int i = 0; i < num_threads; i++)
  {
    args[i].count = &count;
    args[i].n_points = &n_points;
    args[i].num_iterations = num_iterations / num_threads;
  }

  for (int i = 1; i < num_threads; ++i)
  {
    workers[i] = std::thread(calculate_pi, &args[i]);
  }

  calculate_pi(&args[0]);

  for (int i = 1; i < num_threads; i++)
  {
    workers[i].join();
  }

  std::cout << "count: " << count << " of " << n_points << std::endl;
  double pi = 4.0 * (double)count / (double)num_iterations;
  std::cout << "Used " << n_points << " points to estimate pi: " << pi
            << std::endl;

  assert(n_points = num_iterations);
  return 0;
}
