#ifndef _TIME_STAMP_H
#define _TIME_STAMP_H

#include <time.h>
#include <stdio.h>

#ifdef _WIN32
    #pragma warning(disable : 4786)
	#pragma warning(disable : 4996)
	#include <Windows.h>	
#else
	#include <sys/time.h>
#endif

struct timeval DW2time(int dur)
{
	struct timeval rs;
	rs.tv_sec  = dur / 1000;
	rs.tv_usec = (dur % 1000) * 1000;
	return rs;
}

class TimeStamp{
public:
	static void StartTime()
	{
	#ifdef _WIN32
		starttime = DW2time(GetTickCount());
	#else
		gettimeofday(&starttime, NULL);
	#endif
	}

	static void StopTime()
	{
	#ifdef _WIN32
		curtime = DW2time(GetTickCount());
	#else
		gettimeofday(&curtime, NULL);
	#endif
	}

	static void StopTime(const char * str)
	{
		StopTime();
		ElapseTime(str);
	}

	static void ElapseTime(const char * str)
	{
		timeval_subtract(&difference,&curtime,&starttime);
		printf("%s %lld s, %lld us\n", str, (long long)difference.tv_sec, (long long)difference.tv_usec);
	}	

	static void AddTime(struct timeval & totalTime)
	{
		totalTime.tv_sec  += difference.tv_sec;
		totalTime.tv_usec += difference.tv_usec;
		if(totalTime.tv_usec >= 1000000)
		{
			totalTime.tv_sec++;
			totalTime.tv_usec -= 1000000;
		}
	}

private:
	static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) 
	{
	  if (x->tv_usec < y->tv_usec) 
	  {
	    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
	    y->tv_usec -= 1000000 * nsec;
	    y->tv_sec += nsec;
	  }
	  if (x->tv_usec - y->tv_usec > 1000000) 
	  {
	    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
	    y->tv_usec += 1000000 * nsec;
	    y->tv_sec -= nsec;
	  }

	  result->tv_sec = x->tv_sec - y->tv_sec;
	  result->tv_usec = x->tv_usec - y->tv_usec;

	  return x->tv_sec < y->tv_sec;
	}

private:
	static struct timeval starttime,curtime,difference;
};

struct timeval TimeStamp::starttime;
struct timeval TimeStamp::curtime;
struct timeval TimeStamp::difference;

#endif