#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <sys/types.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <unistd.h>
#include <time.h>

struct config {
	unsigned int dummy;
	unsigned int goods;
	unsigned int bids;
	unsigned int * bids_ptr;
	unsigned int * values;
	unsigned int * dummies;
	unsigned int ** alloc2;
};

struct config * get_config(FILE * fp) {
	const char * s_goods = "goods";
	const char * s_bids = "bids";
	const char * s_dummy = "dummy";

	int goods = -1;
	int bids = -1;
	int dummy = -1;
	char * line = NULL;
	size_t len = 0;
	int all = 0;
	while ((getline(&line, &len, fp)) != -1 && !all) {
		if (line[0] == '%' || line[0] == '\n') {
			continue;
		}
		if (strncmp(line, s_goods, strlen(s_goods)) == 0) {
			goods = atoi(line + strlen(s_goods) + 1);
			printf("Number of goods %u\n", goods);
		} else if (strncmp(line, s_bids, strlen(s_bids)) == 0) {
			bids = atoi(line + strlen(s_bids) + 1);
			printf("Number of bids %u\n", bids);
		} else if (strncmp(line, s_dummy, strlen(s_dummy)) == 0) {
			dummy = atoi(line + strlen(s_dummy) + 1);
			printf("Number of dummy goods %u\n", dummy);
		}
		all = !!(goods != -1 && bids != -1 && dummy !=-1);
	}
	free(line);
	if(all == 0) {
		printf("Could not parse file! Exit\n");
		exit(EXIT_FAILURE);
	}
	struct config * ret = (struct config *) malloc(sizeof(struct config));
	ret->goods = goods;
	ret->dummy = dummy;
	ret->bids = bids;
	return ret;
}

void get_bids(FILE * fp, struct config * conf) {

	int goods = -1;
	int bids = -1;
	int dummy = -1;
	char * line = NULL;
	size_t len = 0;
	unsigned int count = 0;

	while ((getline(&line, &len, fp)) != -1) {
		//if the line is not a bid line
		if (!isdigit(line[0])) {
			continue;
		}
		char * head,*tail;
		head = tail = line;
		//get first tab after id
		while(*head != '\t') { head++;}
		tail = head;
		head++;
		//get first tab after value
		while(*head != '\t') { head++;}
		unsigned int value = strtol(tail,&head,10);
		tail = head;
		head++;
		printf("value %u",value);
		unsigned int allocation = 0;
		while(*head != '#') {
			while(*head != '\t') { head++;}
			int good = strtol(tail,&head,10);
			allocation |= (1 << good);
			printf("\t%d",good);
			tail = head;
			head++;
		}
		conf->values[count] = value;
		conf->bids_ptr[count] = allocation;
		count++;
		printf("\n");

	}
	assert(count == conf->bids);

}

int main(int argc, char *argv[]) {

	FILE * fp;
	fp = fopen(argv[1], "r");
	if (fp == NULL) {
		printf("Could not open file %s\n",argv[1]);
		exit(EXIT_FAILURE);
	}
	printf("Reading file %s\n",argv[1]);
	struct config * conf = get_config(fp);
	unsigned int bids = conf->bids;
	if(conf->dummy != -1) {
		conf->dummies = (unsigned int *) malloc(sizeof(unsigned int)*bids);
	}
	conf->values = (unsigned int *) malloc(sizeof(unsigned int)*bids);
	conf->bids_ptr = (unsigned int *) malloc(sizeof(unsigned int)*bids);
	get_bids(fp,conf);
	unsigned int columns = 1+bids/33;
	unsigned int **array2 = malloc(bids * sizeof(int *));
	array2[0] = malloc(bids * columns * sizeof(int));
	int i;
	for(i = 1; i < bids; i++) {
		array2[i] = array2[0] + i * columns;
	}

	for(i = 0; i < bids; i++) {
		int j;
		for(j = 0; j < columns;j++) {
			array2[i][j] = 0;
		}
		for(j = i+1; j < bids; j++) {
			int result = !(conf->bids_ptr[i] & conf->bids_ptr[j]);
			int bit_index = j % 32;
			int word_index = j / 32;
			array2[i][word_index] |= (1 << bit_index);
			printf("%d",result);
		}
		printf("\n");
	}

}
