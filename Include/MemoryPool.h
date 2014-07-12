// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _MEMORY_POOL_H
#define _MEMORY_POOL_H

#include "Common.h"

namespace utils
{

class MemoryPool
{
public:
    MemoryPool()
    {
        m_pMem     = NULL;
        m_nMemSize = 0;
        m_nAvailPhys = GetAvailPhysMemorySize();

        if(m_nAvailPhys < 16*24*24)
            m_nMemMax = m_nAvailPhys >> 1;
        else
            m_nMemMax = m_nAvailPhys - 8*1024*1024;
    }

    virtual ~MemoryPool()
    {
        if(m_pMem)
        {
            delete m_pMem;
            m_pMem = NULL;
        }
    }

public:

    unsigned char* Allocate(uint64_t nFileLen, uint64_t &nAllocatedSize)
    {
        unsigned int nTargetSize;

        if(nFileLen <= m_nMemSize)
        {
            nAllocatedSize = nFileLen;
            return m_pMem;
        }

        nTargetSize = (nFileLen < m_nMemMax) ? nFileLen : m_nMemMax;

        if(m_pMem != NULL)
        {
            delete m_pMem;
            m_pMem = NULL;
            m_nMemSize = 0;
        }

        m_pMem = new unsigned char[nTargetSize];

        if(m_pMem != NULL)
        {
            m_nMemSize     = nTargetSize;
            nAllocatedSize = nTargetSize;
            return m_pMem;
        }
        else
        {
            nAllocatedSize = 0;
            return NULL;
        }
    }

private:
    unsigned char      * m_pMem;
    uint64_t    m_nMemSize;
    uint64_t    m_nMemMax;
    uint64_t    m_nAvailPhys;
};

};

#endif