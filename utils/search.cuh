#ifndef UTILS_SEARCH_CUH
#define UTILS_SEARCH_CUH

#include <cstdint>

template<typename T1, typename T2>
__forceinline__ __device__ T2 lower_bound(const T1* array, const T2 size, const T1 v)
{
    if (array == NULL || size == 0 || array[size - 1] < v) return size;

    T2 low = 0u, high = size - 1, mid = (low + high) / 2;
    while (low < high)
    {
        if (array[mid] < v)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

template<typename T1, typename T2>
__forceinline__ __device__ T2 upper_bound(const T1* array, const T2 size, const T1 v)
{
    T2 count = size, step, first = 0, mid;

    while (count > 0)
    {
        step = count / 2;
        mid = first + step;
        if (array[mid] <= v)
        {
            first = mid + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

#endif