#include <cuda.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>
#include <time.h>

#define SCALE 49000
#define BLOCK_LENGTH 512
#define DIM 512
#define MASK(bit) (~(-1L<<bit))
#define HIHALF(num,bit) num>>(bit/2)
#define LOHALF(num,bit) num&MASK(bit/2)
#define HIGH(num) HIHALF(num,sizeof(num)*8)
#define LOW(num) LOHALF(num,sizeof(num)*8)
#define LONGCAT(hi,low,bit) (((long long)hi<<(bit/2))+(unsigned)low)
#define LCROL(num,bit) ((num&MASK(bit-1))<<1)+((num&(1<<bit-1))!=0)

__constant__ long long CUIP_Table[64] =
{
    1L<<64-58, 1L<<64-50, 1L<<64-42, 1L<<64-34, 1L<<64-26, 1L<<64-18, 1L<<64-10, 1L<<64-2,
    1L<<64-60, 1L<<64-52, 1L<<64-44, 1L<<64-36, 1L<<64-28, 1L<<64-20, 1L<<64-12, 1L<<64-4,
    1L<<64-62, 1L<<64-54, 1L<<64-46, 1L<<64-38, 1L<<64-30, 1L<<64-22, 1L<<64-14, 1L<<64-6,
    1L<<64-64, 1L<<64-56, 1L<<64-48, 1L<<64-40, 1L<<64-32, 1L<<64-24, 1L<<64-16, 1L<<64-8,
    1L<<64-57, 1L<<64-49, 1L<<64-41, 1L<<64-33, 1L<<64-25, 1L<<64-17,  1L<<64-9, 1L<<64-1,
    1L<<64-59, 1L<<64-51, 1L<<64-43, 1L<<64-35, 1L<<64-27, 1L<<64-19, 1L<<64-11, 1L<<64-3,
    1L<<64-61, 1L<<64-53, 1L<<64-45, 1L<<64-37, 1L<<64-29, 1L<<64-21, 1L<<64-13, 1L<<64-5,
    1L<<64-63, 1L<<64-55, 1L<<64-47, 1L<<64-39, 1L<<64-31, 1L<<64-23, 1L<<64-15, 1L<<64-7
};

//逆初始置换表IP^-1
__constant__ long long CUIP_1_Table[64] =
{
    1L<<64-40, 1L<<64-8, 1L<<64-48, 1L<<64-16, 1L<<64-56, 1L<<64-24, 1L<<64-64, 1L<<64-32,
    1L<<64-39, 1L<<64-7, 1L<<64-47, 1L<<64-15, 1L<<64-55, 1L<<64-23, 1L<<64-63, 1L<<64-31,
    1L<<64-38, 1L<<64-6, 1L<<64-46, 1L<<64-14, 1L<<64-54, 1L<<64-22, 1L<<64-62, 1L<<64-30,
    1L<<64-37, 1L<<64-5, 1L<<64-45, 1L<<64-13, 1L<<64-53, 1L<<64-21, 1L<<64-61, 1L<<64-29,
    1L<<64-36, 1L<<64-4, 1L<<64-44, 1L<<64-12, 1L<<64-52, 1L<<64-20, 1L<<64-60, 1L<<64-28,
    1L<<64-35, 1L<<64-3, 1L<<64-43, 1L<<64-11, 1L<<64-51, 1L<<64-19, 1L<<64-59, 1L<<64-27,
    1L<<64-34, 1L<<64-2, 1L<<64-42, 1L<<64-10, 1L<<64-50, 1L<<64-18, 1L<<64-58, 1L<<64-26,
    1L<<64-33, 1L<<64-1, 1L<<64-41,   1L<<64-9, 1L<<64-49, 1L<<64-17, 1L<<64-57, 1L<<64-25
};

//扩充置换表E
__constant__ long long CUE_Table[48] =
{
    1L<<32-32,  1L<<32-1,   1L<<32-2,   1L<<32-3,   1L<<32-4,   1L<<32-5,
    1L<<32-4,   1L<<32-5,   1L<<32-6,   1L<<32-7,   1L<<32-8,   1L<<32-9,
    1L<<32-8,   1L<<32-9, 1L<<32-10, 1L<<32-11, 1L<<32-12, 1L<<32-13,
    1L<<32-12, 1L<<32-13, 1L<<32-14, 1L<<32-15, 1L<<32-16, 1L<<32-17,
    1L<<32-16, 1L<<32-17, 1L<<32-18, 1L<<32-19, 1L<<32-20, 1L<<32-21,
    1L<<32-20, 1L<<32-21, 1L<<32-22, 1L<<32-23, 1L<<32-24, 1L<<32-25,
    1L<<32-24, 1L<<32-25, 1L<<32-26, 1L<<32-27, 1L<<32-28, 1L<<32-29,
    1L<<32-28, 1L<<32-29, 1L<<32-30, 1L<<32-31, 1L<<32-32,   1L<<32-1
};

//置换函数P
__constant__ unsigned int CUP[32] =
{
    1L<<32-16,   1L<<32-7, 1L<<32-20, 1L<<32-21,
    1L<<32-29, 1L<<32-12, 1L<<32-28, 1L<<32-17,
    1L<<32-1, 1L<<32-15, 1L<<32-23, 1L<<32-26,
    1L<<32-5, 1L<<32-18, 1L<<32-31, 1L<<32-10,
    1L<<32-2,   1L<<32-8, 1L<<32-24, 1L<<32-14,
    1L<<32-32, 1L<<32-27,   1L<<32-3,   1L<<32-9,
    1L<<32-19, 1L<<32-13, 1L<<32-30,   1L<<32-6,
    1L<<32-22, 1L<<32-11,   1L<<32-4, 1L<<32-25
};

//S盒
__constant__ const char CUS[8][4][16] = 
{
    // S1
    14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7,
    0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8,
    4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0,
    15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13,
    // S2 
    15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10,
    3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5,
    0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15,
    13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9,
    // S3 
    10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8,
    13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1,
    13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7,
    1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12,
    // S4 
    7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15,
    13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9,
    10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4,
    3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14,
    // S5 
    2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9,
    14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6,
    4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14,
    11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3,
    // S6 
    12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11,
    10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8,
    9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6,
    4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13,
    // S7 
    4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1,
    13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6,
    1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2,
    6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12,
    // S8 
    13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7,
    1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2,
    7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8,
    2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11
};

//置换选择1
__constant__ long long PC_1[56] =
{
    1L<<64-57,1L<<64-49,1L<<64-41,1L<<64-33,1L<<64-25,1L<<64-17,1L<<64-9,
    1L<<64-1,1L<<64-58,1L<<64-50,1L<<64-42,1L<<64-34,1L<<64-26,1L<<64-18,
    1L<<64-10,1L<<64-2,1L<<64-59,1L<<64-51,1L<<64-43,1L<<64-35,1L<<64-27,
    1L<<64-19,1L<<64-11,1L<<64-3,1L<<64-60,1L<<64-52,1L<<64-44,1L<<64-36,
    1L<<64-63,1L<<64-55,1L<<64-47,1L<<64-39,1L<<64-31,1L<<64-23,1L<<64-15,
    1L<<64-7,1L<<64-62,1L<<64-54,1L<<64-46,1L<<64-38,1L<<64-30,1L<<64-22,
    1L<<64-14,1L<<64-6,1L<<64-61,1L<<64-53,1L<<64-45,1L<<64-37,1L<<64-29,
    1L<<64-21,1L<<64-13,1L<<64-5,1L<<64-28,1L<<64-20,1L<<64-12,1L<<64-4
};

//置换选择2
__constant__ long long PC_2[48] =
{
    1L<<56-14,1L<<56-17,1L<<56-11,1L<<56-24,1L<<56-1,1L<<56-5,
    1L<<56-3,1L<<56-28,1L<<56-15,1L<<56-6,1L<<56-21,1L<<56-10,
    1L<<56-23,1L<<56-19,1L<<56-12,1L<<56-4,1L<<56-26,1L<<56-8,
    1L<<56-16,1L<<56-7,1L<<56-27,1L<<56-20,1L<<56-13,1L<<56-2,
    1L<<56-41,1L<<56-52,1L<<56-31,1L<<56-37,1L<<56-47,1L<<56-55,
    1L<<56-30,1L<<56-40,1L<<56-51,1L<<56-45,1L<<56-33,1L<<56-48,
    1L<<56-44,1L<<56-49,1L<<56-39,1L<<56-56,1L<<56-34,1L<<56-53,
    1L<<56-46,1L<<56-42,1L<<56-50,1L<<56-36,1L<<56-29,1L<<56-32
};
__constant__ int MOVE_TIMES[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};

__device__ float getnextrand(curandState *state)
{
  return (float)(curand_uniform(state));
}

__device__ long long getnextrandscaled(curandState *state, int scale)
{
  return (long) scale * getnextrand(state)*scale;
}

__global__ void initCurand(curandState *state)
{
    int idx = threadIdx.x;
    curand_init(idx, 0, 0, &state[idx]);
}


__device__ long long des_cuda_applypc(long long key,long long*pc,int len)
{
    long long ret=0;
    ret=(key&pc[0])!=0?1:0;
    for(int i=1; i<len; i++)
    {
        ret<<=1;
        ret+=(key&pc[i])!=0?1:0;
    }
    return ret;
}

__device__ int des_cuda_applyPBox(int key)
{
    int ret=0;
    unsigned int *pc = CUP;
    int i=0;
    ret=(key&pc[0])!=0?1:0;
    for(i=1; i<32; i++)
    {
        ret<<=1;
        ret+=(key&pc[i])!=0?1:0;
    }
    return ret;
}

__device__ long long des_lrol(int *hi,int *low,int loop)
{
    while(loop>0)
    {
        *hi = LCROL(*hi,28);
        *low = LCROL(*low,28);
        loop--;
    }
    long long ret = LONGCAT(*hi,*low,56);
    return ret;
}

__device__ int des_ckey(long long key,long long *store)//生成轮换key
{
    long long tmp = des_cuda_applypc(key,PC_1,56);
    int hi = HIHALF(tmp,56);
    int low = LOHALF(tmp,56);
    for(int i=0; i<16; i++)
    {
        tmp = des_lrol(&hi,&low,MOVE_TIMES[i]);
        store[i]=des_cuda_applypc(tmp,PC_2,48);
    }
    return 0;
}

__device__ int des_cuda_applySBox(long long data)
{
    int tmp,part,i,row,col;
    tmp=0;
    for(i=0; i<8; i++)
    {
        row = ((data&0x20)>>4)+(data&0x1);
        col = (data&0x1E)>>1;
        data>>=6;
        part=CUS[7-i][row][col];
        part<<=(4*i);
        tmp+=part;
    }
    return tmp;
}

__device__ long long des_cuda_one(long long data,long long key)
{
    int hi=HIGH(data);
    int low=LOW(data);
    long long tmp;int sub;
    tmp = des_cuda_applypc(low,CUE_Table,48);
    tmp^=key;
    sub = des_cuda_applySBox(tmp);
    sub = des_cuda_applyPBox(sub);
    sub ^=hi;tmp = LONGCAT(low,sub,64);
    return tmp;
}

__global__ void des_cuda(curandState*state,long long *data)
{
    //int idx = threadIdx.x;
    int idx=blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx>=BLOCK_LENGTH*DIM) return; 
    long long t, ckey[16];int tim=0;
    while(tim<=(1<<7))
    {
        long long key= getnextrandscaled(&state[idx], SCALE);
        des_ckey(key,ckey);t = des_cuda_applypc(data[idx],CUIP_Table,64);
        for(int i=0;i<16;i++) t = des_cuda_one(t,ckey[i]);
        int h=HIGH(t),l=LOW(t);t = LONGCAT(l,h,64);
        data[idx] = des_cuda_applypc(t,CUIP_1_Table,64);
        tim++;
    }
}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

typedef long long int64;

int main() 
{
    curandState *devState;
    cudaMalloc((void**)&devState, DIM*BLOCK_LENGTH * sizeof(curandState));
    cudaCheckErrors("cudamalloc1");
    initCurand<<<DIM,BLOCK_LENGTH>>>(devState); 
    cudaDeviceSynchronize();
    cudaCheckErrors("cudamalloc2");

    int64*data=new int64[BLOCK_LENGTH*DIM];int size=DIM*BLOCK_LENGTH*sizeof(int64);
    for(int i=0;i<BLOCK_LENGTH*DIM;i++) data[i]=rand();
    int64*d_data;cudaMalloc((void**)&d_data,size);
    cudaMemcpy(d_data,data,size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudamalloc3");
    struct timeval tstart, tend;
    gettimeofday(&tstart, NULL);    
    des_cuda<<<DIM,BLOCK_LENGTH>>>(devState,d_data);
    cudaDeviceSynchronize();
    cudaCheckErrors("cudamalloc4");

    cudaMemcpy(data,d_data,size,cudaMemcpyDeviceToHost);
    gettimeofday(&tend, NULL);
    long long uses = 1000000 * (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec);
    printf("loop time: %lld\n", uses);  
    cudaCheckErrors("cudamalloc5");
    for(int i=0;i<10;i++) printf("i: %d value: %lld\n",i,data[i]);
}
