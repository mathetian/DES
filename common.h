#ifndef _RAINBOW_CHAIN_H
#define _RAINBOW_CHAIN_H

#include <stdio.h>
typedef struct{
	int nStartIndex;
	int nEndIndex;
}RainbowChain;

inline unsigned int GetFileLen(FILE*file)
{
}

inline void Logo()
{
	printf("DESRainbowCuda 1.0 - Make an implementation of DES Time-and-Memory Tradeoff Technology\n");
	printf("by Tian Yulong(mathetian@gmail.com)\n");
}

inline unsigned int GetAvailPhysMemorySize()
{
}
#endif