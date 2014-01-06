#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <sys/time.h>
#include <openssl/des.h>

class RainbowChain{
public:
	uint64_t nStartKey, nEndKey;
	bool operator < (const RainbowChain &m) const 
	{
        return nEndKey < m.nEndKey;
    }
};

extern unsigned int GetFileLen(FILE*file);

extern void Logo();


extern unsigned int GetAvailPhysMemorySize();

extern void U56ToArr7(const uint64_t & key56, unsigned char * key_56);


extern void Arr7ToU56(const unsigned char * key_56, uint64_t & key56);
/**
	des_cblock: typedef unsigned char DES_cblock[8]
**/

extern void SetupDESKey(const uint64_t&key56,des_key_schedule &ks);

extern bool AnylysisFileName(const char * filename, uint64_t & chainLen, uint64_t & chainCount);

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
	static FILE * file;
	static FILE * tmpFile;

private:
	int offset, length, curOffset;
	RainbowChain chains[CHAIN_IN_MEMORY_MAX];
};

#endif