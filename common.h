#ifndef _RAINBOW_CHAIN_H
#define _RAINBOW_CHAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <openssl/des.h>

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

inline void U56ToArr7(const uint64_t & key56, unsigned char * key_56)
{	
	int mask = (1<<8) - 1;
	
	key_56[0] = key56 & mask;
	key_56[1] = (key56 >>  8) & mask;
	key_56[2] = (key56 >> 16) & mask;
	key_56[3] = (key56 >> 24) & mask;
	key_56[4] = (key56 >> 32) & mask;
	key_56[5] = (key56 >> 40) & mask;
	key_56[6] = (key56 >> 48) & mask;
}

inline void Arr7ToU56(const unsigned char * key_56, uint64_t & key56)
{
	int index; key56 = 0;
	for(index = 0;index < 7;index++)
		key56 |= (key_56[index] << (8*index));
}
/**
	des_cblock: typedef unsigned char DES_cblock[8]
**/

inline void SetupDESKey(const uint64_t&key56,des_key_schedule &ks)
{
	des_cblock key, key_56;
	
	U56ToArr7(key56,key_56);

	key[0]=key_56[0];
	key[1]=(key_56[0]<<7)|(key_56[1]>>1);
	key[2]=(key_56[1]<<6)|(key_56[2]>>2);
	key[3]=(key_56[2]<<5)|(key_56[3]>>3);
	key[4]=(key_56[3]<<4)|(key_56[4]>>4);
	key[5]=(key_56[4]<<3)|(key_56[5]>>5);
	key[6]=(key_56[5]<<2)|(key_56[6]>>6);
	key[7]=(key_56[6<<1]);

	des_set_key(&key,ks);
}

#endif