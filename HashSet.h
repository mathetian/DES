#ifndef _HASH_SET_H
#define _HASH_SET_H
class HashSet{
public:
	HashSet();
	virtual ~ HashSet();
private:
	vector<string> m_vHash;
	vector<string> m_vFound;
	vector<string> m_vPlain;
	vector<string> m_vBinary;
public:
	void AddHash(string sHash);
	void AnyHashLeft();
	void AnyHashLeftWithLen();
	void GetLeftHashWithLen();
	void SetPlain(string sHash);
	int GetStatHashFound();
	int GetStatHashTotal();
};
#endif