//////////////////////////////////
/* Stores a full or partial Bid */
//////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bid.h"

/* constructor */
Bid::Bid(double a, int i)
{
  //vec = v;
  amount = a;
  array = new unsigned[array_size = 10];
  num_goods = 0;  // the number of goods belonging to this bid
  visited = false;
  sorted = false;
  indexInBidSet = i;
  conflicts = 0;
}

/* destructor */
Bid::~Bid()
{
  delete[] array;
}

int Bid::firstDummyGood(unsigned total_goods)
{
  for (int t=num_goods-1;t>=0; t--)
    if (array[t] >= total_goods) return array[t];
  return -1;
}

void Bid::addGood(unsigned g)
{
  if (num_goods == array_size)
    {
      unsigned *temp_array = new unsigned[array_size *= 2];
      if(temp_array == 0)
	{
	  fprintf(stderr, "Bid::addGood: Out of memory\n");
	  exit(1);
	}

      for (int t=0;t<num_goods;t++)
	temp_array[t] = array[t];
      delete[] array;
      array = temp_array;
    }

  // when not in debug mode, optimizer will remove this loop
  for (int i=0; i<num_goods; i++)
    assert(array[i] != g);

  array[num_goods++] = g;
  sorted = false;
}

// log-time implementation on ordered bid list
int Bid::indexOf(unsigned g) {

  if (!sorted) sort();

  if (num_goods == 0) return -1;
	
  // jump is a power of two representing how far we will jump for the next try
  // it is initialized to the largest power of 2 that's <= num_goods
  int jump = 1;
  while (jump <= num_goods) jump *= 2;
  jump /= 2;
	
  int guess = jump-1;
	
  while (array[guess] != g) {
    jump /= 2;
    if (jump == 0) return -1;
    if (array[guess] < g) {
      while (guess + jump >= num_goods) {
	jump /= 2;
	if (jump == 0) return -1;
      }
      guess += jump;
    }
    else
      guess -= jump;
  }
	
  return guess;
}

bool Bid::conflictsWith(Bid *other) {
	
  if (!sorted) sort();
  if (!other->sorted) other->sort();

  int myIt = 0, otherIt = 0;
	
  while (myIt < num_goods && otherIt < other->num_goods) {
    if (array[myIt] < other->array[otherIt])
      myIt++;
    else if (array[myIt] > other->array[otherIt])
      otherIt++;
    else
      return true;
  }
	
  return false;
}

bool Bid::subsetEqualOf(Bid *other)
{

  // bid must not be longer
  if (num_goods > other->num_goods) return false;

  if (!sorted) sort();
  if (!other->sorted) other->sort();

  int myIt = 0, otherIt = 0;
	
  while (myIt < num_goods) {
    if (otherIt >= other->num_goods || array[myIt] < other->array[otherIt])
      return false;
    else if (array[myIt] > other->array[otherIt])
      otherIt++;
    else
      myIt++;
  }
	
  return true;
}

int Bid::largestGood() {
  if (sorted)
    return array[num_goods-1];
  else {
    int largest = -1;
    for (int i=0; i<num_goods; i++)
      if ((int)array[i] > largest)
	largest = array[i];
    return largest;
  }
}

void Bid::renumber(int *newNums) {
  for (int i=0; i<num_goods; i++)
    array[i] = newNums[array[i]];
}


void Bid::sort() {
  if (sorted) return;
	
  unsigned *tempArray = new unsigned[num_goods];
  sortFrom(0, num_goods-1, tempArray);
  delete[] tempArray;
	
  sorted = true;
}

void Bid::sortFrom(int begin, int end, unsigned *tempArray) {
  if (begin < end) {
    int middle = (begin + end) / 2;
    sortFrom (begin, middle, tempArray);
    sortFrom (middle + 1, end, tempArray);
    merge (begin, middle + 1, end, tempArray);
  }
}

void Bid::merge(int leftBegin, int rightBegin, int rightEnd, unsigned *tempArray) {
  int leftEnd = rightBegin-1;
  int tempPos = leftBegin;
	
  int leftPos = leftBegin, rightPos = rightBegin;
  while (leftPos <= leftEnd && rightPos <= rightEnd)
    if (array[leftPos] < array[rightPos] )
      tempArray[tempPos++] = array[leftPos++];
    else
      tempArray[tempPos++] = array[rightPos++];
		
  while (leftPos <= leftEnd)
    tempArray[tempPos++] = array[leftPos++];
		
  while (rightPos <= rightEnd)
    tempArray[tempPos++] = array[rightPos++];
		
  for (int i=leftBegin; i<=rightEnd; i++)
    array[i] = tempArray[i];
}

