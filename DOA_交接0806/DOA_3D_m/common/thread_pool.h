#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#define MAX_THREADS 16
#define THREAD_TASK_QUEUE_SIZE 128

// 任務結構定義
typedef struct {
    void (*function)(void*);
    void *arg;
} Thread_Task;

typedef struct {
    int thread_index;
} ThreadArg;

void bind_main_to_core(int core_id);

void init_thread_pool(int num_threads, int start_core);

void addThreadTask(void (*func)(void*), void *arg);

void wait_for_all_tasks();

void destroy_thread_pool();

int get_thread_index();
#endif // THREAD_POOL_H
