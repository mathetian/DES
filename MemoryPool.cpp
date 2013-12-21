#include "MemoryPool.h"

MemoryPool::MemoryPool()
{
	m_pMem=NULL;
	m_nMemSize=0;
	unsigned int nAvailPhys=GetAvailPhysMemorySize();
	if(nAvailPhys<16*24*24)
		m_nMemMax=nAvailPhys/2;
	else
		m_nMemMax=nAvailPhys-8*1024*1024;
}

MemoryPool::~MemoryPool()
{
	if(m_pMem)
	{
		delete m_pMem;
		m_pMem=0;
		m_nMemSize=0;
	}
}

unsigned char* MemoryPool::Allocate(unsigned int nFileLen,unsigned int&nAllocatedSize)
{
	if(nFileLen<=m_nMemSize)
	{
		nAllocatedSize=nFileLen;
		return m_pMem;
	}

	unsigned int nTargetSize;
	if(nFileLen<m_nMemMax)
		nTargetSize=nFileLen;
	else 
		nTargetSize=m_nMemMax;
	if(m_pMem!=NULL)
	{
		delete m_pMem;
		m_pMem=NULL;
		m_nMemSize=0;
	}
	m_pMem=new unsigned char[nTargetSize];
	if(m_pMem!=NULL)
	{
		m_nMemSize=nTargetSize;
		nAllocatedSize=nTargetSize;
		return m_pMem;
	}
	else
	{
		nAllocatedSize=0;
		return NULL;
	}
}