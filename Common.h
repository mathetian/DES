#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <sys/time.h>
#include <openssl/des.h>

typedef struct _RainbowChain{
	int nStartKey, nEndKey;
	bool operator < (const struct _RainbowChain &m) const 
	{
        return nStartKey < m.nStartKey;
    }
}RainbowChain;

extern unsigned int GetFileLen(FILE*file);

extern void Logo();


extern unsigned int GetAvailPhysMemorySize();

extern void U56ToArr7(const uint64_t & key56, unsigned char * key_56);


extern void Arr7ToU56(const unsigned char * key_56, uint64_t & key56);
/**
	des_cblock: typedef unsigned char DES_cblock[8]
**/

extern void SetupDESKey(const uint64_t&key56,des_key_schedule &ks);

#define CHAIN_IN_MEMORY_MAX 1024 

class SortedSegment{
public:
	SortedSegment();
	virtual ~ SortedSegment();

public:
	RainbowChain * getFirst();
	RainbowChain * getAll();
	RainbowChain * getNext();
	void setProperty(int offset,int length,int curOffset);
	int  getLength();

public:
	 FILE * file;
	 FILE * tmpFile;

private:
	int offset, length, curOffset;
	RainbowChain chains[CHAIN_IN_MEMORY_MAX];
};

#endif