#ifndef _MEMORY_POOL_H
#define _MEMORY_POOL_H

class MemoryPool{
public:
	MemoryPool();
	virtual ~ MemoryPool();
private:
	unsigned char* m_pMem;
	unsigned int m_nMemSize;
	unsigned int m_nMemMax;
public:
	unsigned char*Allocate(unsigned int nFileLen,unsigned int&nAllocateSize);
};
#endif