#include "HashAlgorithm.h"

void setupDESKey(unsigned char*key_56,des_key_schedule &ks,int nKeyLen)
{
	des_cblock key;
	if(nKeyLen==7)
	{
		key[0]=key_56[0];
		key[1]=(key_56[0]<<7)|(key_56[1]>>1);
		key[2]=(key_56[1]<<6)|(key_56[2]>>2);
		key[3]=(key_56[2]<<5)|(key_56[3]>>3);
		key[4]=(key_56[3]<<4)|(key_56[4]>>4);
		key[5]=(key_56[4]<<3)|(key_56[5]>>5);
		key[6]=(key_56[5]<<2)|(key_56[6]>>6);
		key[7]=(key_56[6<<1]);
		des_set_key(&key,ks);
	}
	else des_set_key(key_56,ks);
}

void HashDES(unsigned char*key,int nKeyLen,unsigned char*pHash)
{
	des_key_schedule ks;setupDESKey(key,ks,nKeyLen);
	static unsigned char magic[]={0x4B,0x47,0x53,0x21,0x40,0x23,0x24,0x25};
	des_ecb_encrypt((des_cblock*)magic,(des_cblock*)pHash,ks,DES_ENCRYPT);
}