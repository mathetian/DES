#include <iostream>
using namespace std;

typedef unsigned long long uint64_t;

int main()
{
	FILE * file = fopen("tt.txt","wb");
	int a = 1000;
	const char str[4] = {0x01,0x02,0x03,0x04};
	a = *(int*)str;
	uint64_t b = 1031606382730;
	cout << sizeof(uint64_t) << endl;
	fwrite((char*)&b,sizeof(uint64_t),1,file);
	//fwrite((char*)&a,sizeof(int),1,file);
	fclose(file);
}