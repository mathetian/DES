#include "common.h"

/**
    des_cblock: typedef unsigned char DES_cblock[8];
**/

des_cblock plainText={0x30,0x55,0x32,0x28,0x6D,0x6F,0x29,0x5A};

void desEncrypt(unsigned char*key)
{
    des_key_schedule keys;des_cblock out;
    int i;for(i=0;i<(1<<10);i++)
    {
        des_cblock keyBlock;memcpy(keyBlock,key,8);
        DES_set_key_unchecked(&keyBlock,&keys);
        memset(out,0,8);
        des_ecb_encrypt(&plainText,&out,keys,DES_ENCRYPT);
        memcpy(key,out,8);
    }
}

void desCrypt(int rank, int numproc) 
{
    struct timeval tstart, tend;unsigned char key[8];
    int round;FILE*f1,*f2;int i,j;char str[30];
    round=0;srand(rank);char fileName[30];
    sprintf(fileName,"start-%d.in",rank);
    if((f1=fopen(fileName,"w"))==NULL)
    {
        printf("desCrypt: fopen start.in error\n");
        exit(0);
    }
    sprintf(fileName,"end-%d.in",rank);
    if((f2=fopen(fileName,"w"))==NULL)
    {
        printf("desCrypt: fopen end.out error\n");
        exit(0);
    }
    printf("%d Starting DES kernel\n",rank);
    while(1)
    {
        printf("%d Begin Round: %d\n",rank,round);
        fprintf(f1,"Begin Round: %d\n",round);
        fprintf(f2,"Begin Round: %d\n",round);
        gettimeofday(&tstart, NULL);
        
        for(i=0;i<(1<<10);i++)
        {
            for(j=0;j<8;j++) key[i]=rand()%256;
            sprintf(str,"0x%x%x%x%x%x%x%x%x\n",key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
            fputs(str,f1);
            desEncrypt(key);
            sprintf(str,"0x%x%x%x%x%x%x%x%x\n",key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
            fputs(str,f2);
        }

        gettimeofday(&tend, NULL);
        int64 uses=1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
        printf("%d round time: %lld us\n",rank,uses);
        fprintf(f1,"round time: %lld us\n",uses);
        fprintf(f2,"round time: %lld us\n",uses);
        
        printf("%d End Round: %d\n",rank,round);
        fprintf(f1,"End Round: %d\n",round);
        fprintf(f2,"End Round: %d\n",round);

        round++;
    }
    fclose(f1);fclose(f2);
    printf("Ending DES kernel\n");
}
    
