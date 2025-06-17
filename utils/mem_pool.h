#ifndef UTILS_MEM_POOL_H_
#define UTILS_MEM_POOL_H_

#include <cstdint>
#include "utils/cuda_helpers.h"

template<typename T>
struct MemPool
{
    T *array_;
    unsigned long long int capacity_;
    unsigned long long int h_occupy_;
    unsigned long long int *occupy_;

    void Alloc(unsigned long long int size)
    {
        cudaErrorCheck(cudaMalloc(&array_, sizeof(T) * size));
        capacity_ = size;
        h_occupy_ = 0ul;
        cudaErrorCheck(cudaMalloc(&occupy_, sizeof(unsigned long long int)));
        cudaErrorCheck(cudaMemset(occupy_, 0u, sizeof(unsigned long long int)));
    }
    void Free()
    {
        cudaErrorCheck(cudaFree(array_));
    }
    void Reset()
    {
        cudaErrorCheck(cudaMemset(occupy_, 0u, sizeof(unsigned long long int)));
    }
    bool OutOfMemory()
    {
        cudaErrorCheck(cudaMemcpy(&h_occupy_, occupy_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        return h_occupy_ >= capacity_;
    }
};

template<typename T>
class CyclicQueue {
  public:
    T *array_;
    unsigned long long int capacity_;
    unsigned long long int available_start_;
    unsigned long long int available_end_;

    void Alloc(size_t capacity) {
        cudaErrorCheck(cudaMalloc(&array_, sizeof(T) * capacity));
        capacity_ = capacity;
        available_start_ = 0;
        available_end_ = capacity;
    }

    void Free() {
        cudaErrorCheck(cudaFree(array_));
    }

    void Reset() {
        available_start_ = 0;
        available_end_ = capacity_;
    }

    unsigned long long int TryMax() {
        return available_start_;
    }

    size_t GetFree() {
        return (available_end_ + capacity_ - available_start_) % capacity_;
    }

    void Push(size_t size) {
        available_start_ = (available_start_ + size) % capacity_;
    }

    void Pop(size_t size) {
        available_end_ = (available_end_ + size) % capacity_;
    }

    T* CopyToHost() {
        T *h_array;
        cudaErrorCheck(cudaMallocHost(&h_array, sizeof(T) * capacity_));
        cudaErrorCheck(cudaMemcpy(h_array, array_, sizeof(T) * capacity_, cudaMemcpyDeviceToHost));
        return h_array;
    }

    T* CopyToHost(unsigned long long int start, unsigned long long int size) {  // gli945
        T *h_array;
        cudaErrorCheck(cudaMallocHost(&h_array, sizeof(T) * size));
        if (start + size > capacity_) {
            cudaErrorCheck(cudaMemcpy(h_array, array_ + start, sizeof(T) * (capacity_ - start), cudaMemcpyDeviceToHost));
            cudaErrorCheck(cudaMemcpy(h_array + capacity_ - start, array_, sizeof(T) * (start + size - capacity_), cudaMemcpyDeviceToHost));
        } else{
            cudaErrorCheck(cudaMemcpy(h_array, array_ + start, sizeof(T) * size, cudaMemcpyDeviceToHost));
        }
        return h_array;
    }
};

#endif  // UTILS_MEM_POOL_H_