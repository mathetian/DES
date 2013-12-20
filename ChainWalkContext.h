#ifndef _CHAIN_WALK_CONTEXT_H
#define _CHAIN_WALK_CONTEXT_H
#include <string>
#include <stdlib.h>
#include <stdint.h>
#include <openssl/rand.h>
using namespace std;

typedef long long uint64;

#define HASH_LEN 8

class ChainWalkContext{
public:
	ChainWalkContext();
	virtual ~ ChainWalkContext();
private:
	static string m_plainText;
	static int m_chainLen;
	static int m_chainCount;
private:
	uint64 m_nIndex;
	unsigned char m_hash[HASH_LEN];
private:
	static void setPlainText(const string&plainText);
	static void setChainLen(int chainLen);
	static void setChainCount(int chainCount);
public:
	static void setProperty(const string&plainText,int chainlen,int chainCount);
	static void dump();
public:
	void GenerateRandomIndex();
	void SetIndex(uint64 nIndex);
	void SetHash(unsigned char*pHash);

	void IndexToPlain();
	void PlainToHash();
	void HashToIndex(int nPos);

	uint64 GetIndex();
	string GetPlain();
	string GetBinary();
	string GetHash();
};
#endif