#ifndef _HASH_ALG_H
#define _HASH_ALG_H

#include <openssl/des.h>

extern void setupDESKey(unsigned char key_56[],des_key_schedule &ks);
extern void HashDES(unsigned char*key,int nPlainLen,unsigned char*pHash);

#endif