#include "DESCommon.h"

bool RainbowChain::operator < (const RainbowChain &m) const
{
    return nEndKey < m.nEndKey;
}

void Logo()
{
    printf("DESRainbowCrack 1.0\n 	Make an implementation of DES Time-and-Memory Tradeoff Technology\n 	By Tian Yulong(mathetian@gmail.com)\n\n");
}

uint64_t GetFileLen(FILE* file)
{
    uint64_t pos = _ftelli64(file);
    _fseeki64(file, 0, SEEK_END);
    uint64_t len = _ftelli64(file);
    
    _fseeki64(file, pos, SEEK_SET);

    return len;
}

uint64_t GetAvailPhysMemorySize()
{
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof (statex);
    GlobalMemoryStatusEx (&statex);
    return statex.ullAvailPhys;
#else
    struct sysinfo info;
    sysinfo(&info);
    return info.freeram;
#endif
}

void U56ToArr7(const uint64_t & key56, unsigned char * key_56)
{
    int mask = (1<<8) - 1;

    key_56[0] = (key56 & mask);
    key_56[1] = ((key56 >>  8) & mask);
    key_56[2] = ((key56 >> 16) & mask);
    key_56[3] = ((key56 >> 24) & mask);
    key_56[4] = ((key56 >> 32) & mask);
    key_56[5] = ((key56 >> 40) & mask);
    key_56[6] = ((key56 >> 48) & mask);
    key_56[7] = ((key56 >> 56) & mask);
}

/**
	Problem with that, how to convert 64 bit wrong to 56 bit right
**/
void Arr7ToU56(const unsigned char * key_56, uint64_t & key56)
{
    key56 = *(uint64_t*)key_56;
}

void SetupDESKey(const uint64_t & key56,des_key_schedule & ks)
{
    des_cblock key_56;

    U56ToArr7(key56, key_56);

    DES_set_key_unchecked(&key_56, &ks);
}

bool AnylysisFileName(const char * filename, uint64_t & chainLen, uint64_t & chainCount)
{
    int len = strlen(filename), i = 0, j;
    if(len <= 6 || filename[3] != '_') return false;
    char str[256];
    memset(str, 0, sizeof(str));
    for(i = 4; i< len; i++) if(filename[i] == '-') break;
    if(i == len || i == 3) return false;
    memcpy(str,filename + 4, i - 4);

    chainLen = atoll(str);

    memset(str, 0, sizeof(str));
    for(j = i + 1; j < len; j++) if(filename[j] == '_') break;
    if(j == len || j == i+1) return false;
    memcpy(str,filename + i + 1,j - i - 1);

    chainCount = atoll(str);

    return true;
}
