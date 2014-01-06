#include "ChainWalkContext.h"

#include <time.h>
#include <stdlib.h>

int main()
{
	ChainWalkContext cwc; uint64_t flag = (1ull << 20) - 1;
	srand(time(NULL));

	int index = 0;
	for(;index < 10;index++)
	{
		uint64_t r = rand() & flag;
		printf("%lld %llu\n",(long long)r, (unsigned long long)cwc.Crypt(r));
	}
	return 0;
}