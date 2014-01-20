#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <openssl/des.h>

#include <sstream>
#include <iostream>
using namespace std;

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;


#ifdef _WIN32
    #pragma warning(disable : 4786)
    #pragma warning(disable : 4996)
    #pragma warning(disable : 4267)
    #pragma warning(disable : 4244)
    #include <Windows.h>        
#else
    #include <sys/sysinfo.h>
    #include <sys/time.h>
#endif

#ifdef _WIN32
    inline uint64_t atoll(const char * str)
    {
            uint64_t rs;
            istringstream ist(str);
            ist >> rs;

            return rs;
    }
#endif


class RainbowChain{
public:
	uint64_t nStartKey, nEndKey;
	bool operator < (const RainbowChain &m) const;
};

extern uint64_t GetFileLen(FILE*file);

extern void Logo();


extern uint64_t GetAvailPhysMemorySize();

extern void U56ToArr7(const uint64_t & key56, unsigned char * key_56);


extern void Arr7ToU56(const unsigned char * key_56, uint64_t & key56);
/**
	des_cblock: typedef unsigned char DES_cblock[8]
**/

extern void SetupDESKey(const uint64_t&key56, des_key_schedule &ks);

extern bool AnylysisFileName(const char * filename, uint64_t & chainLen, uint64_t & chainCount);

#define CHAIN_IN_MEMORY_MAX 1024 

/*class SortedSegment{
public:
	SortedSegment();
	virtual ~ SortedSegment();

public:
	RainbowChain * getAll();
	RainbowChain * getNext();
	void setProperty(int offset,int length,int curOffset);
	int  getLength();

public:
	FILE * tmpFile;

private:
	int offset, length, curOffset;
	RainbowChain chains[CHAIN_IN_MEMORY_MAX];
};
<<<<<<< HEAD
*/
#endif