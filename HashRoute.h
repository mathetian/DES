#ifndef _HASH_ROUTE_H
#define _HASH_ROUTE_H
#include <string>
#include <vector>
using namespace std;

typedef void (*HASHROUTINE)(unsigned char*pPlain,unsigned char*key,unsigned char*pHash);

class HashRoute{
public:
	HashRoute();
	virtual ~ HashRoute();
private:
	vector<string> vHashRoutineName;
	vector<HASHROUTINE> vHashRoutine;
	vector<int> vHashLen;
	void AddHashRoutine(string sHashRoutineName,HASHROUTINE&pHashRoutine,int nHashLen);
public:
	string GetHashRoutine(string sHashRoutineName,HASHROUTINE&pHashRoutine,int&nHashLen);
};

#endif