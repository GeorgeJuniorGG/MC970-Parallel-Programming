#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS 5

// Print hello world from a thread
void *printHello(void *threadId) {
  long tid = (long)threadId;

  printf("Hello Worlds from thread #%ld!\n", tid);

  // Pause for 3 seconds
  sleep(3);

  printf("Goodbye from thread #%ld!\n", tid);

  return NULL;
}

int main() {
  long t;

  for (t = 0; t < NUM_THREADS; t++) {
    printf("Creating thread #%ld\n", t);
    printHello((void *)t);
  }

  // Wait for all threads to finish
  for (t = 0; t < NUM_THREADS; t++) {
  }
}