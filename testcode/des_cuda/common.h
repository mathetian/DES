#ifndef common_h
#define common_h

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <openssl/evp.h>
#include <sys/time.h>

#define OUTPUT_QUIET		0
#define OUTPUT_NORMAL		1
#define OUTPUT_VERBOSE		2

#define NUM_BLOCK_PER_MULTIPROCESSOR	3
#define SIZE_BLOCK_PER_MULTIPROCESSOR	256*1024
#define MAX_THREAD			256


#define STATE_THREAD_DES	2
#define DES_MAXNR		8
#define DES_BLOCK_SIZE		8
#define DES_KEY_SIZE		8
#define DES_KEY_SIZE_64		8

#include <openssl/evp.h>
typedef struct cuda_crypt_parameters_st {
	const unsigned char *in;	// host input buffer 
	unsigned char *out;		// host output buffer
	size_t nbytes;			// number of bytes to be operated on
	EVP_CIPHER_CTX *ctx;		// EVP OpenSSL structure
	uint8_t *host_data;		// possible page-locked host memory
	uint64_t *d_in;		// Device memory (input)
	uint64_t *d_out;		// Device memory (output)
} cuda_crypt_parameters;


// Split uint64_t into two uint32_t and convert each from BE to LE
#define nl2i(s,a,b)      a = ((s >> 24L) & 0x000000ff) | \
			     ((s >> 8L ) & 0x0000ff00) | \
			     ((s << 8L ) & 0x00ff0000) | \
			     ((s << 24L) & 0xff000000),   \
			 b = ((s >> 56L) & 0x000000ff) | \
			     ((s >> 40L) & 0x0000ff00) | \
			     ((s >> 24L) & 0x00ff0000) | \
			     ((s >> 8L) & 0xff000000)

// Convert uint64_t endianness
#define flip64(a)	(a= \
			(a << 56) | \
			((a & 0xff00) << 40) | \
			((a & 0xff0000) << 24) | \
			((a & 0xff000000) << 8)  | \
			((a & 0xff00000000) >> 8)  | \
			((a & 0xff0000000000) >> 24) | \
			((a & 0xff000000000000) >> 40) | \
			(a >> 56))

#ifndef TX
	#if (__CUDA_ARCH__ < 200)
		#define TX (__umul24(blockIdx.x,blockDim.x) + threadIdx.x)
	#else
		#define TX (blockIdx.x * blockDim.x + threadIdx.x)
	#endif
#endif

#define _CUDA(call) {																	\
	call;				                                												\
	cudaerrno=cudaGetLastError();																	\
	if(cudaSuccess!=cudaerrno) {                                       					         						\
		if (output_verbosity!=OUTPUT_QUIET) fprintf(stderr, "Cuda error %d in file '%s' in line %i: %s.\n",cudaerrno,__FILE__,__LINE__,cudaGetErrorString(cudaerrno));	\
		exit(EXIT_FAILURE);                                                  											\
    } }

static int __attribute__((unused)) output_verbosity;
static int __attribute__((unused)) isIntegrated;
#endif