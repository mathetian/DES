#include <algorithm>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
using namespace std;

#include "DESCommon.h"
#include "TimeStamp.h"
#include "DESChainWalkContext.h"

#include <assert.h>

void Usage()
{
    Logo();
    printf("Usage  : sort sort number fileName\n");
    printf("         sort distinct number fileName\n");
    printf("example 1: sort sort 1 DES_100_100_test\n");
    printf("example 2: sort distinct 1 DES_100_100_test\n\n");
}

typedef pair<RainbowChain, int> PPR;

/**fix cmp bug**/
struct cmp
{
    bool operator()(PPR a,PPR b)
    {
        RainbowChain  r1 = a.first;
        RainbowChain  r2 = b.first;
        if(r1.nEndKey < r2.nEndKey)
            return true;
        return false;
    }
};

void QuickSort(RainbowChain * pChain, uint64_t length)
{
    sort(pChain, pChain + length);
}

void ExternalSort(FILE * file, vector <FILE*> tmpFiles)
{
    int index = 0;
    RainbowChain chain;

    fseek(file, 0, SEEK_SET);

    vector <uint64_t> tmpLens(tmpFiles.size(), 0);

    int ss = (int)tmpFiles.size();

    for(; index < ss; index++)
    {
        fseek(tmpFiles[index], 0, SEEK_SET);
        tmpLens[index] = GetFileLen(tmpFiles[index]) >> 4;
    }

    priority_queue<PPR, vector<PPR>, cmp> chainPQ;

    for(index = 0; index < ss; index++)
    {
        assert(fread((char*)&chain,sizeof(RainbowChain),1,tmpFiles[index]) == 1);
        chainPQ.push(make_pair(chain,index));
    }

    while(!chainPQ.empty())
    {
        chain = chainPQ.top().first;
        index = chainPQ.top().second;

        chainPQ.pop();

        fwrite((char*)&chain, sizeof(RainbowChain), 1, file);
        tmpLens[index]--;
        if(tmpLens[index] == 0) continue;
        assert(fread((char*)&chain, sizeof(RainbowChain), 1, tmpFiles[index]) == 1);

        chainPQ.push(make_pair(chain, index));
    }
}

void ExternalSort(FILE * file)
{
    uint64_t nAvailPhys, fileLen;

    uint64_t memoryCount;
    int tmpNum;
    int index = 0;

    char str[256];

    nAvailPhys = GetAvailPhysMemorySize();

    fileLen    = GetFileLen(file);

    memoryCount = nAvailPhys >> 4;

    uint64_t eachLen = memoryCount << 4;
    uint64_t lastLen = fileLen % eachLen;
    if(lastLen == 0) lastLen = eachLen;

    tmpNum      = fileLen/nAvailPhys;

    if(fileLen % nAvailPhys != 0) tmpNum++;

    assert((nAvailPhys <= fileLen) && "Error ExternalSort type\n");

    RainbowChain * chains =  (RainbowChain*)new unsigned char[eachLen];

    fseek(file, 0, SEEK_SET);

    vector <FILE*> tmpFiles(tmpNum, NULL);

    for(; index < tmpNum; index++)
    {
        sprintf(str,"tmpFiles-%d",index);
        tmpFiles[index] = fopen(str, "wb");
        assert(tmpFiles[index] &&("tmpFiles fopen error\n"));
        if(index < tmpNum - 1)
        {
            assert(fread((char*)chains, sizeof(RainbowChain), memoryCount, file) == memoryCount);
            QuickSort(chains, memoryCount);
            fwrite((char*)chains, sizeof(RainbowChain), memoryCount, tmpFiles[index]);
        }
        else
        {
            assert(fread((char*)chains, lastLen, 1, file) == 1);
            assert((lastLen % 16 == 0) && ("Error lastLen"));
            QuickSort(chains, lastLen >> 4);
            fwrite((char*)&chains, lastLen, 1, tmpFiles[index]);
        }
    }

    ExternalSort(file, tmpFiles);

    for(index = 0; index < tmpNum; index++)
    {
        fclose(tmpFiles[index]);
    }
}

void printMemory(const char * str, long long nAvailPhys)
{
    long long a = 1000, b = 1000*1000;
    long long c = b * 1000;
    printf("%s %lld GB, %lld MB, %lld KB, %lld B\n", str, nAvailPhys/c, (nAvailPhys%c)/b, (nAvailPhys%b)/a, nAvailPhys%1000);
}

void Distinct(const char * fileName)
{
    FILE * file;
    uint64_t fileLen;

    if((file = fopen(fileName, "rb")) == NULL)
    {
        printf("Failed to open: %s\n",fileName);
        return;
    }

    fileLen = GetFileLen(file);

    uint64_t nRainbowChainCount = fileLen >> 4;

    RainbowChain * pChain = (RainbowChain*)new unsigned char[fileLen];
    RainbowChain * tmpChain = (RainbowChain*)new unsigned char[fileLen];

    fseek(file, 0, SEEK_SET);

    printf("Begin Read file\n");
    if(fread(pChain, 1, fileLen, file) != fileLen)
    {
        printf("disk read fail\n");
        return;
    }
    printf("End Read file\n");
    fclose(file);

    printf("Begin Distinct\n");
    uint64_t index = 0, num =0;
    while(index < nRainbowChainCount)
    {
        tmpChain[num++] = pChain[index];
        while(index + 1 < nRainbowChainCount && \
                pChain[index].nEndKey == pChain[index+1].nEndKey)
            index++;
        index ++;
    }
    FILE * file2 = fopen("Distinct.txt","wb");
    assert(file2 && "fopen error");
    printf("End Distinct\n");
    fwrite((char*)tmpChain, sizeof(RainbowChain), num, file2);

    cout<<GetFileLen(file2)<<" "<<GetFileLen(file2)/16<<endl;
    fclose(file2);
}

void SortFiles(vector <string> fileNames, vector <FILE*> files, const char * prefix)
{
    int index = 0;
    uint64_t nAvailPhys;
    char str[256];

    vector <uint64_t> fileLens(fileNames.size(), 0);

    FILE * targetFile;

    nAvailPhys = GetAvailPhysMemorySize();
    sprintf(str, "Available free physical memory: ");
    printMemory(str, nAvailPhys);

    int ss = (int)fileNames.size();

    for(; index < ss; index++)
    {

        uint64_t & fileLen = fileLens[index];

        fileLen = GetFileLen(files[index]);

        assert((fileLen % 16 ==0) && ("Rainbow table size check failed\n"));

        printf("%s FileLen %lld bytes\n", fileNames[index].c_str(), (long long)fileLen);

        if(nAvailPhys > fileLen)
        {
            uint64_t nRainbowChainCount = fileLen >> 4;

            RainbowChain * pChain = (RainbowChain*)new unsigned char[fileLen];

            if(pChain!=NULL)
            {
                printf("%d, Loading rainbow table...\n", index);

                fseek(files[index], 0, SEEK_SET);

                if(fread(pChain, 1, fileLen, files[index]) != fileLen)
                {
                    printf("%d, disk read fail\n", index);
                    goto ABORT;
                }

                printf("%d, Sorting the rainbow table...\n", index);

                QuickSort(pChain, nRainbowChainCount);

                printf("%d, Writing sorted rainbow table...\n", index);

                fseek(files[index], 0, SEEK_SET);
                fwrite(pChain, 1, fileLen, files[index]);
                delete [] pChain;
            }
        }
        else ExternalSort(files[index]);
    }

    targetFile = fopen(prefix,"wb");
    fclose(targetFile);

    targetFile = fopen(prefix,"rb+");
    assert(targetFile && ("targetFile fopen error\n"));

    printf("Begin Actually ExternalSort\n");
    ExternalSort(targetFile, files);
    printf("End Actually ExternalSort\n");
    fclose(targetFile);

ABORT:
    for(index = 0; index < ss; index++)
        fclose(files[index]);

}

void SortOneFile(const char * prefix)
{
    uint64_t nAvailPhys;
    char str[256];

    FILE * targetFile;
    uint64_t fileLen;

    nAvailPhys = GetAvailPhysMemorySize();
    sprintf(str, "Available free physical memory: ");
    printMemory(str, nAvailPhys);

    targetFile = fopen(prefix,"rb+");
    assert(targetFile && "fopen error\n");

    fileLen = GetFileLen(targetFile);

    assert((fileLen % 16 ==0) && ("Rainbow table size check failed\n"));

    printf("%s FileLen %lld bytes\n", prefix, (long long)fileLen);

    if(nAvailPhys > fileLen)
    {
        uint64_t nRainbowChainCount = fileLen >> 4;

        RainbowChain * pChain = (RainbowChain*)new unsigned char[fileLen];

        if(pChain != NULL)
        {
            _fseeki64(targetFile, 0, SEEK_SET);

            if(fread(pChain, 1, fileLen, targetFile) != fileLen)
            {
                printf("disk read fail %d\n");
                goto ABORT;
            }

            printf("Sorting the rainbow table...\n");

            QuickSort(pChain, nRainbowChainCount);

            printf("Writing sorted rainbow table...\n");

            _fseeki64(targetFile, 0, SEEK_SET);
            assert(fwrite(pChain, 1, fileLen, targetFile)==fileLen);

            delete [] pChain;
        }
    }
    else ExternalSort(targetFile);

ABORT:
    fclose(targetFile);
}

void SortLargeFile(const char * prefix)
{
    char str[256];

    FILE * file;
    uint64_t fileLen, nAvailPhys;

    nAvailPhys = GetAvailPhysMemorySize();
    sprintf(str, "Available free physical memory: ");
    printMemory(str, nAvailPhys);

    file = fopen(prefix,"rb+");
    assert(file && "fopen error\n");

    fileLen = GetFileLen(file);

    uint64_t memoryCount;
    int tmpNum;
    int index = 0;

    memoryCount = nAvailPhys >> 4;

    uint64_t eachLen = memoryCount << 4;
    uint64_t lastLen = fileLen % eachLen;
    if(lastLen == 0) lastLen = eachLen;

    tmpNum      = fileLen/nAvailPhys;

    if(fileLen % nAvailPhys != 0) tmpNum++;

    assert((nAvailPhys <= fileLen) && "Error ExternalSort type\n");

    RainbowChain * chains =  (RainbowChain*)new unsigned char[eachLen];

    fseek(file, 0, SEEK_SET);

    vector <FILE*> tmpFiles(tmpNum, NULL);

    for(; index < tmpNum; index++)
    {
        sprintf(str,"tmpFiles-%d",index);
        tmpFiles[index] = fopen(str, "wb");
        assert(tmpFiles[index] &&("tmpFiles fopen error\n"));
        if(index < tmpNum - 1)
        {
            assert(fread((char*)chains, sizeof(RainbowChain), memoryCount, file) == memoryCount);
            QuickSort(chains, memoryCount);
            fwrite((char*)chains, sizeof(RainbowChain), memoryCount, tmpFiles[index]);
        }
        else
        {
            assert(fread((char*)chains, lastLen, 1, file) == 1);
            assert((lastLen % 16 == 0) && ("Error lastLen"));
            QuickSort(chains, lastLen >> 4);
            fwrite((char*)&chains, lastLen, 1, tmpFiles[index]);
        }
    }

    ExternalSort(file, tmpFiles);

    for(index = 0; index < tmpNum; index++)
    {
        fclose(tmpFiles[index]);
    }
}

#define TTWO (1048576*32)

void verifiedSorted(const char *fileName)
{
    FILE *file = fopen(fileName,"rb+");
    _fseeki64(file, 0, SEEK_SET);
    assert(_ftelli64(file) == 0);

    RainbowChain chains[TTWO];TimeStamp tms1;
    TimeStamp tms;struct timeval totalTime; totalTime.tv_sec = 0;
    totalTime.tv_usec = 0; char str[256];tms1.StartTime();
    for(uint64_t i=0;i<4;i++)
    {  
        tms.StartTime();
        assert(fread((char*)&chains, sizeof(RainbowChain), TTWO, file) == TTWO);
        assert(_ftelli64(file) == TTWO*(i+1)*16);
        sprintf(str,"round%d: ", (int)i);
        tms.StopTime(str);

        tms.StartTime();
        QuickSort(chains, TTWO);
        tms.StopTime();
        sprintf(str, "round-%d.data",i);
        tms.StopTime(str);
        tms.StartTime();
        FILE *file1 = fopen(str,"wb+"); 
        assert(file1);
        assert(fwrite((char*)&chains[0],  sizeof(RainbowChain), TTWO, file1) == TTWO);
        fclose(file1);
        tms.StopTime("write time: ");
       //tms.AddTime(totalTime);
        // for(uint64_t j=0;j<TTWO - 1;j++)
        // {
        //     assert(chains[i*TTWO+j].nEndKey < chains[i*TTWO+j+1].nEndKey);
        // }
    }
    tms1.StopTime("totalTime: ");
}

uint64_t BinarySearch(RainbowChain * pChain, uint64_t pChainCount, uint64_t nIndex)
{
    long long low=0, high=pChainCount;
    while(low<high)
    {
        long long mid = (low+high)/2;
        if(pChain[mid].nEndKey == nIndex) return mid;
        else if(pChain[mid].nEndKey < nIndex) low = mid + 1;
        else high=mid;
    }
    return low;
}

int TestCollision()
{
    char str[256]; RainbowChain chains[TTWO];
    uint64_t i, m_nIndex;
    RAND_bytes((unsigned char*)&m_nIndex,5);
    printf("%lld \n",(long long)m_nIndex);
    TimeStamp tms; TimeStamp tms1;
    tms.StartTime();
    for(i = 0;i < 4;i++)
    {
        sprintf(str,"round-%d.data",(int)i);
        printf("Begin read file: %s\n", str);
        tms1.StartTime();
        FILE *file = fopen(str,"rb+");
        assert(file);
        _fseeki64(file, 0, SEEK_SET);
        assert(_ftelli64(file) == 0);
        assert(fread((char*)&chains[0], sizeof(RainbowChain), TTWO, file) == TTWO);
        fclose(file);
        tms1.StopTime("Read time: ");
        tms1.StartTime();
        uint64_t rs = BinarySearch(chains, TTWO, m_nIndex);
        tms1.StopTime("BinarySearch time: ");
        if(chains[rs].nEndKey == m_nIndex)
        {
            printf("found in file %s\n",str);
            break;
        }
    }
   tms.StopTime("totalTime: ");
    if(i==4) return 0;
    return 1;
}   

int TestCollision2()
{
    char str[256]; 
    RainbowChain chains[TTWO];
    uint64_t chain2[TTWO];
    TimeStamp tms; TimeStamp tms1;

    uint64_t i, j, m_nIndex;

    tms.StartTime();
    for(i = 0;i<TTWO;i++)
    {
        RAND_bytes((unsigned char*)&m_nIndex,5);
        chain2[i] = m_nIndex;
    }
    tms.StopTime("Generate time: ");

    tms.StartTime();
    for(i = 0;i < 4;i++)
    {
        sprintf(str,"round-%d.data",(int)i);
        printf("Begin read file: %s\n", str);
        tms1.StartTime();
        FILE *file = fopen(str,"rb+");
        assert(file);
        _fseeki64(file, 0, SEEK_SET);
        assert(_ftelli64(file) == 0);
        assert(fread((char*)&chains[0], sizeof(RainbowChain), TTWO, file) == TTWO);
        fclose(file);
        tms1.StopTime("Read time: ");
        tms1.StartTime();int times = 0;
        for(j=0;j<TTWO;j++)
        {
            uint64_t rs = BinarySearch(chains, TTWO, chain2[j]);
            if(chains[rs].nEndKey == chain2[j])
            {
                times++;
                if(times%100 == 0) cout<<"found:"<<j<<"time"<<times<<endl;
               // printf("found in file %s %d %lld\n",str,(int)j,(long long)chain2[j]);
            }
        }
        tms1.StopTime("BinarySearch time: ");
    }
   tms.StopTime("totalTime: ");
    if(i==4) return 0;
    return 1;
}   


int main(int argc,char*argv[])
{
    // {
    //     verifiedSorted(argv[1]);
    //     return 0;
    // }
    // {
    //     for(uint64_t index = 0;index < 10;index++)
    //     {
    //         if(TestCollision() ==1) 
    //         {
    //             printf("index %d\n",index);
    //         }
    //         if(index %10==0)
    //             printf("round %d\n",(int)index);
    //     }
    //     return 0;
    // }
    // {
    //     TestCollision2();
    // }
    if(argc != 4)
    {
        Usage();
        return 0;
    }
    
    if(strcmp(argv[1],"distinct") == 0)
    {
        int num = atoi(argv[2]);
        assert((num == 1) && "Sorry for that, I want to write `less` code");
        Distinct(argv[3]);
    }
    else if(strcmp(argv[1],"sort") == 0)
    {
        int num =  atoi(argv[2]);

        assert((num < 9) && (num >= 1) && ("sorry number must be less than ten and more than zero\n"));

        if(num == 1)
        {
            printf("Begin Sort One File\n");
            SortOneFile(argv[3]);
            printf("End Sort One File\n");
        }
        else
        {
            vector <string> fileNames(num, "");
            vector <FILE*>  files(num, NULL);

            for(int index = 0; index < num; index++)
            {
                fileNames[index] = argv[3];
                fileNames[index] +=  "_";
                fileNames[index].push_back(index + '0');
                files[index] = fopen(fileNames[index].c_str(),"rb+");
                assert(files[index] && "fopen error\n");
            }

            printf("Begin SortFiles\n");
            SortFiles(fileNames, files, argv[3]);
            printf("End SortFiles\n");
        }
    }
    else Usage();

    return 0;
}