#include <stdio.h>
#include <omp.h>

int main()
{
  omp_set_num_threads(4);
  double arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  double maxVal = 0.0;
  double minVal = 100.0;
  float avg = 0.0, sum = 0.0, sumVal = 0.0;
  int i;

#pragma omp parallel for reduction(min : minVal)
  for (i = 0; i < 10; i++)
  {
    printf("thread id = %d and i = %d \n", omp_get_thread_num(), i);
    if (arr[i] < minVal)
    {
      minVal = arr[i];
    }
  }
  printf("minVal = %f", minVal);
  printf("\n");
  printf("\n");

#pragma omp parallel for reduction(max : maxVal)
  for (i = 0; i < 10; i++)
  {
    printf("thread id = %d and i = %d \n", omp_get_thread_num(), i);
    if (arr[i] > maxVal)
    {
      maxVal = arr[i];
    }
  }
  printf("maxVal = %f", maxVal);
  printf("\n");
  printf("\n");

#pragma omp parallel for reduction(+ : sumVal)
  for (i = 0; i < 10; i++)
  {
    printf("thread id = %d and i = %d \n", omp_get_thread_num(), i);
    sumVal = sumVal + arr[i];
  }
  printf("sumVal = %f", sumVal);
  printf("\n");
  printf("\n");

#pragma omp parallel for reduction(+ : sum)
  for (i = 0; i < 10; i++)
  {
    printf("thread id = %d and i = %d \n", omp_get_thread_num(), i);
    sum = sum + arr[i];
  }
  avg = sum / 10;
  printf("avg_val = %f", avg);
  printf("\n");
  printf("\n");

}


//output

// thread id = 0 and i = 0 
// thread id = 2 and i = 2 
// thread id = 3 and i = 3 
// thread id = 1 and i = 1 
// thread id = 0 and i = 4 
// thread id = 2 and i = 6 
// thread id = 3 and i = 7 
// thread id = 1 and i = 5 
// thread id = 0 and i = 8 
// thread id = 2 and i = 9 
// minVal = 1.000000

// thread id = 1 and i = 1 
// thread id = 0 and i = 0 
// thread id = 3 and i = 3 
// thread id = 2 and i = 2 
// thread id = 1 and i = 5 
// thread id = 3 and i = 7 
// thread id = 0 and i = 4 
// thread id = 2 and i = 6 
// thread id = 0 and i = 8 
// thread id = 2 and i = 9 
// maxVal = 10.000000

// thread id = 1 and i = 1 
// thread id = 3 and i = 3 
// thread id = 0 and i = 0 
// thread id = 2 and i = 2 
// thread id = 1 and i = 5 
// thread id = 3 and i = 7 
// thread id = 2 and i = 6 
// thread id = 0 and i = 4 
// thread id = 2 and i = 9 
// thread id = 0 and i = 8 
// sumVal = 55.000000

// thread id = 3 and i = 3 
// thread id = 1 and i = 1 
// thread id = 0 and i = 0 
// thread id = 2 and i = 2 
// thread id = 1 and i = 5 
// thread id = 0 and i = 4 
// thread id = 2 and i = 6 
// thread id = 3 and i = 7 
// thread id = 0 and i = 8 
// thread id = 2 and i = 9 
// avg_val = 5.500000