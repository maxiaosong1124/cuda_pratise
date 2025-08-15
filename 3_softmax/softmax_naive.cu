#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

void softmax_cpu(float* d_in, float* d_out, int N, int C)
{ 
    for(int i = 0; i < N; ++i)
    {
        const float* in_row = d_in + i * C;
        float* out_row = d_out + i * C;
        float maxval = -INFINITY;
        for(int j = 0; j < C; ++j)
        {
            if(maxval < in_row[j])
            {
                maxval = in_row[j];
            }
        }
        float sum = 0.f;
        for(int j = 0; j < C; ++j)
        {
            out_row[j] = expf(in_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / float(sum);
        for(int j = 0; j < C; ++j)
        {
            out_row[j] *= norm;
        }
    }
}
