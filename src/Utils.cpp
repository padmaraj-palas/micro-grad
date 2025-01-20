#include <stdlib.h>
#include <time.h>

#include "Utils.h"

void initUtils()
{
	srand(time(NULL));
}

double getRandomBetween(const double& min, const double& max)
{
	double randomNumer = (double)rand() / RAND_MAX;
	return min * (1 - randomNumer) + max * randomNumer;
}