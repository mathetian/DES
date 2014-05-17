#include "DESCipherSet.h"

#include <iostream>
using namespace std;

DESCipherSet * DESCipherSet::p_cs;

DESCipherSet * DESCipherSet::GetInstance()
{
    if(!p_cs)
        p_cs = new DESCipherSet();
    return p_cs;
}

DESCipherSet::DESCipherSet() : index(0), solve(0)
{
}

DESCipherSet::~DESCipherSet()
{
}

void DESCipherSet::AddKey(uint64_t cipherKey)
{
    m_vKeys.push_back(cipherKey);
}

bool DESCipherSet::AnyKeyLeft()
{
    return index == m_vKeys.size() ? 0 : 1;
}

uint64_t DESCipherSet::GetLeftKey()
{
    solve = 0;
    return m_vKeys.at(index);
}

void DESCipherSet::AddResult(uint64_t cipherKey, uint64_t key)
{
    m_maps[cipherKey].push_back(key);
}

void DESCipherSet::Succeed()
{
    solve = 1;
}

void DESCipherSet::Done(uint64_t cipherKey)
{
    index++;
}

bool DESCipherSet::Solved()
{
    return solve == 1 ? 1 : 0;
}

int DESCipherSet::GetKeyFoundNum()
{
    return m_vFound.size();
}

void DESCipherSet::PrintAllFound()
{
}

int DESCipherSet::Detect(RainbowChain chain)
{
    vector<uint64_t> tmp = m_maps[chain.nEndKey];
    if(tmp.size() == 0) return 0;
    cout<<chain.nEndKey<<" "<<tmp.size()<<endl;
    for(uint64_t i=0;i<tmp.size();i++)
    {
        if(tmp.at(i) ==  chain.nStartKey)
            return 1;
    }
    return 0;
}