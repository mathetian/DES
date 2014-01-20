#include <iostream>
#include <fstream>
using namespace std;

#include <stdio.h>
#include <assert.h>

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

int main()
{
	FILE * file = fopen("data","r");
	FILE * file2 = fopen("data2","w");
	assert(file && file2);
	uint64_t a,b;
	while(fscanf(file,"%d%d",&a,&b))
	{
		cout <<a<<" "<<b<<endl;
		
		uint32_t a1 = a & ((1ull << 32) - 1);
		uint32_t a2 = (a >> 32);

		uint32_t b1 = b & ((1ull << 32) - 1);
		uint32_t b2 = (b >> 32);

		fwrite((unsigned char*)&a1,sizeof(uint32_t),1,file2);
		fwrite((unsigned char*)&a2,sizeof(uint32_t),1,file2);
		fwrite((unsigned char*)&b1,sizeof(uint32_t),1,file2);
		fwrite((unsigned char*)&b2,sizeof(uint32_t),1,file2);
	}
}