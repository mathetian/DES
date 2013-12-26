#include <dirent.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>

#define CHECKEXE(mode) mode & 00100
#define CHECKW(mode)   mode & 00200
#define CHECKR(mode)   mode & 00400

/**
    S_IRUSR(S_IREAD) 00400     文件所有者具可读取权限
    S_IWUSR(S_IWRITE)00200     文件所有者具可写入权限
    S_IXUSR(S_IEXEC) 00100     文件所有者具可执行权限
**/

int main(int argc, char *argv[])
{
	DIR*dp;struct dirent*dirp;int count;
	if ((dp = opendir(".")) == NULL)
	{
		printf("error open\n");
		return 0;
	}
	count = 0;
	while ((dirp = readdir(dp)) != NULL)
	{	
		struct stat buf;
    	stat(dirp -> d_name, &buf);
    	if(CHECKEXE(buf.st_mode) && S_ISDIR(buf.st_mode) == 0)
    	{
			printf("%d: %s\n", count++,dirp -> d_name);
			remove(dirp -> d_name);
    	}
		/*if(access(dirp->d_name,X_OK) == 0&&dirp -> d_name[0] != '.')
		{
			printf("%d: %s\n", count++,dirp -> d_name);
			remove(dirp -> d_name);
		}	
		*/	
	}
	closedir(dp);
	return 0;
}