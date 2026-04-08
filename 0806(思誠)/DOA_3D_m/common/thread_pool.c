#include "thread_pool.h"
#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>


// 任務佇列
static Thread_Task taskQueue[THREAD_TASK_QUEUE_SIZE];
static int head = 0;
static int tail = 0;
static int count = 0;

// 多執行緒控制
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_t threads[MAX_THREADS];
static pthread_t thread_id_map[MAX_THREADS];
static int shutdown_pool = 0;
static int pending_tasks = 0;
static pthread_cond_t all_done = PTHREAD_COND_INITIALIZER;

// main core
void bind_main_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t main_thread = pthread_self();
    int rc = pthread_setaffinity_np(main_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        fprintf(stderr, "Error binding main thread to core %d\n", core_id);
    } else {
        printf("Main thread bound to core %d\n", core_id);
    }
}
// Worker function
static void* thread_worker(void *arg) {
    ThreadArg* t_arg = (ThreadArg*)arg;
    int my_index = t_arg->thread_index;
    free(t_arg);  // 用完釋放

    pthread_mutex_lock(&lock);
    thread_id_map[my_index] = pthread_self();  // 記錄對應的 thread ID
    pthread_mutex_unlock(&lock);

    //printf("Thread %d registered. ID = %lu\n", my_index, thread_id_map[my_index]);


    while (1) {
        pthread_mutex_lock(&lock);

        while (count == 0 && !shutdown_pool) {
            pthread_cond_wait(&cond, &lock);
        }

        // 收到關閉訊號就結束 thread
        if (shutdown_pool && count == 0) {
            pthread_mutex_unlock(&lock);
            break;
        }

        // 取出任務
        Thread_Task task = taskQueue[head];
        head = (head + 1) % THREAD_TASK_QUEUE_SIZE;
        count--;

        pthread_mutex_unlock(&lock);

        // 執行任務
        task.function(task.arg);

        // 執行完任務後更新 pending_tasks
        pthread_mutex_lock(&lock);
        pending_tasks--;
        if (pending_tasks == 0) {
            // 任務都做完，喚醒等待的主執行緒
            pthread_cond_signal(&all_done);
        }
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}



// init ThreadPool
void init_thread_pool(int num_threads, int start_core) {
    int total_cores = sysconf(_SC_NPROCESSORS_ONLN);

    for (int i = 0; i < num_threads; i++) {
        ThreadArg* arg = malloc(sizeof(ThreadArg));
        arg->thread_index = i;
        pthread_create(&threads[i], NULL, thread_worker, arg);

        // 設定 CPU affinity：從 start_core 開始輪流綁定
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        int core_id = (start_core + i) % total_cores;  // 避免超出 CPU 編號範圍
        CPU_SET(core_id, &cpuset);

        int rc = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            fprintf(stderr, "Error setting thread affinity for thread %d\n", i);
        } else {
            printf("Thread %d created with pthread_t = %lu, bound to CPU core %d\n",
                   i, threads[i], core_id);
        }
    }
}


void addThreadTask(void (*func)(void*), void *arg) {
    pthread_mutex_lock(&lock);
    if (count < THREAD_TASK_QUEUE_SIZE) {
        taskQueue[tail].function = func;
        taskQueue[tail].arg = arg;
        tail = (tail + 1) % THREAD_TASK_QUEUE_SIZE;
        count++;
        pending_tasks++;  // 每加一個任務就 +1
        pthread_cond_signal(&cond);
    } else {
        fprintf(stderr, "Task queue is full!\n");
    }
    pthread_mutex_unlock(&lock);
}

void wait_for_all_tasks() {
    pthread_mutex_lock(&lock);
    while (pending_tasks > 0) {
        pthread_cond_wait(&all_done, &lock);
    }
    pthread_mutex_unlock(&lock);
}

// 結束 ThreadPool（不強制中斷，等任務做完）
void destroy_thread_pool() {
    pthread_mutex_lock(&lock);
    shutdown_pool = 1;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);

    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}


int get_thread_index() {
    pthread_t self_id = pthread_self();
    for (int i = 0; i < MAX_THREADS; i++) {
        if (pthread_equal(self_id, thread_id_map[i])) {
            return i;
        }
    }
    return -1; // 找不到
}