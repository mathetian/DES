#ifndef _CIPHER_SET_H
#define _CIPHER_SET_H

#include <vector>
using namespace std;

#include "DESCommon.h"

class DESCipherSet
{
public:
    ~ DESCipherSet();
    static DESCipherSet * GetInstance();
public:
    void     AddKey(uint64_t cipherKey);
    bool     AnyKeyLeft();
    uint64_t GetLeftKey();

    void     AddResult(uint64_t cipherKey,uint64_t key);
    void     Done();
    void 	 Succeed();
    bool     Solved();
    int      GetKeyFoundNum();
    void     PrintAllFound();

private:
    DESCipherSet();
    static DESCipherSet * p_cs;

    vector<uint64_t> m_vKeys;
    vector<pair<uint64_t,uint64_t> > m_vFound;
    uint32_t index;
    int solve;
};

#endif