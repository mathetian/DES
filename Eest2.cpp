#include <openssl/des.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#include "Common.h"

unsigned char m_dplainText[8] = {0x30,0x55,0x32,0x28,0x6D,0x6F,0x29,0x5A};

int main()
{
	uint64_t key = 933728;
	des_key_schedule ks;unsigned char out[8];
	SetupDESKey(key,ks);memset(out,0,8);
	des_ecb_encrypt(&m_dplainText,&out,ks,DES_ENCRYPT);
	Arr7ToU56(out, key);
	cout << key << endl;
	return 0;
}