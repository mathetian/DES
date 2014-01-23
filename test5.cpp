#include <stdio.h>
#include <openssl/des.h>
#include <iostream>
using namespace std;

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
	unsigned char dd[8]={0xFE,0xFE,0xFE,0xFE,0xFE,0xFE,0xFE,0xFE};
	DES_key_schedule sch;
	DES_set_key_unchecked(&dd,&sch);
	for(int i=0;i<16;i++)
	{
		unsigned long a = *((unsigned long*)&(sch.ks[i].cblock));
		cout<<a<<endl;
	}
}