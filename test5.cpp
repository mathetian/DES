#include <iostream>
using namespace std;

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <openssl/des.h>

/*typedef unsigned int DES_LONG;

typedef unsigned char const_DES_cblock[8];

void fun(const_DES_cblock * key)
{
	register const unsigned char *in;
	in = &(*key)[0];
	for(int i=0;i<8;i++)
	{
		int a=(int)in[i];
		printf("%d\n",a);
	}
}*/

int main()
{
	/*const_DES_cblock dd= {1,3,4,5,6,7,8,9};
	fun(&dd);*/
	//unsigned char dd[8]={0x0E,0x0E,0x0E,0x0E,0x0E,0x0E,0x0E,0x0E};
	//unsigned char dd[8]={0xFE,0xFE,0xFE,0xFE,0xFE,0xFE,0xFE,0xFE};
	unsigned char dd[8] = {0x0E,0x0E,0x0E,0x0E,0x0E,0x0E,0x0E,0x02};
	DES_key_schedule sch;
	DES_set_key_unchecked(&dd,&sch);
	for(int i=0;i<16;i++)
	{
		unsigned long a = *((unsigned long*)&(sch.ks[i].cblock));
		cout<<a<<endl;
	}

	uint64_t plain=0x6D6F295A30553228;
	//unsigned char * in = (unsigned char*)&plain;
	//unsigned char in[8] = {0x28,0x32,0x55,0x30,0x5A,0x29,0x6F,0x6D};
	//unsigned char in[8] = {0x5A,0x29,0x6F,0x6D,0x5A,0x29,0x6F,0x6D};
	unsigned char in[8] = {0x28,0x32,0x55,0x30,0xFF,0xFF,0xFF,0xFF};
	unsigned char out[8];
	memset(out,0,8);
	des_ecb_encrypt(&in,&out,sch,DES_ENCRYPT);
	uint64_t ci=*(uint64_t*)out;
	uint64_t ci2=*(uint64_t*)in;
	cout<<ci<<" "<<ci2<<endl;
}