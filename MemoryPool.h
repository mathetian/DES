#ifndef _MEMORY_POOL_H
#define _MEMORY_POOL_H
#include "common.h"

class MemoryPool{
public:
	MemoryPool();
	virtual ~ MemoryPool();

public:
	unsigned char*Allocate(unsigned int nFileLen,unsigned int&nAllocateSize);

private:
	unsigned char * m_pMem;
	unsigned int    m_nMemSize;
	unsigned int    m_nMemMax;
	unsigned int    m_nAvailPhys;
};
#endif