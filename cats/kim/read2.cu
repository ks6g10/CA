/**
 em	 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
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
#include <cuda_runtime_api.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


typedef uint32_t myint;
#define WORDSIZE (sizeof(myint)*8)
#define SIZE (1)
struct allocation {
	myint a[SIZE];
};

struct bid {
	struct allocation alloc;
	unsigned int id;
	unsigned int bin;
	unsigned int dummy;
	unsigned int offer;
	float average;
	struct bid * next;
	struct bid * prev;
};

struct configuration {
	struct allocation * allocation;
	unsigned char * bin;
	unsigned int * id;
	// each bids dummy, if any else is set to 0
	unsigned int * dummies;
	unsigned int * bin_count;
	unsigned short * max_offset;
	unsigned int * offer;
	float * average;
	// the allocation dummy
	unsigned int goods;
	unsigned int words;
	unsigned int bids;
	unsigned int dummy;
	unsigned int singletons;
	unsigned int * allocation_id;
	unsigned int * allocation_dummy;
	unsigned int allocation_id_index;
	unsigned short * bin_index;
	float allocation_value;
};

unsigned int ints = 0;
/* unsigned int * MASK = void; */

unsigned int next_index(unsigned int a_index) {
	return __builtin_ffs(a_index) - 1;
//	tmp  &= ~(1 << index);
//	upper_bound +=  v(bins[x],pi_conf,bin_counts[x]);

}

void print_binary(unsigned int * allocation, unsigned int goods) {
	int x;
	for (x = goods - 1; x >= 0; x--) {
		printf("%u", !!(*allocation & (1 << x)));
	}
	printf("\n");
}

struct configuration * get_configuration(FILE * fp) {
	const char * s_goods = "goods";
	const char * s_bids = "bids";
	const char * s_dummy = "dummy";

	char * line = NULL;
	size_t len = 0;
	int got_dummy = 0;
	unsigned int all = 0;
	unsigned int goods = 0;
	unsigned int bids = 0;
	unsigned int dummy = 0;

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
			got_dummy = 1;
			printf("Number of dummy %u\n", dummy);
//			ints = 1+(goods-1)/32;
		}
//			total_goods = goods + dummy;
		all = !!(goods && bids && got_dummy);
	}
	free(line);

	//if(goods <= 32) {
	struct configuration * ret = (struct configuration *) malloc(sizeof(struct configuration));
	ret->goods = goods;
	ret->dummy = dummy;
	ret->bids = bids;

	//}

	ret->words = SIZE;
	return ret;
}

unsigned int * get_bincount(FILE * fp, struct configuration * conf, unsigned int * have_singelton) {
	unsigned int * bin_count = (unsigned int *) malloc(sizeof(int) * conf->goods);
	int x;
	for (x = 0; x < conf->goods; x++) {
		bin_count[x] = 0;
	}

	char * head = NULL;
	char * tail = NULL;
	char * line = NULL;
	size_t len = 0;
	printf("hello1\n");
	while ((getline(&line, &len, fp)) != -1) {
		if (!isdigit(line[0])) {
			continue;
		}
		head = tail = line;
		int tab_count = 0;
		head++;
		while (tab_count < 3) {
			if (*head == '\t') {
				tab_count++;
				if (tab_count <= 2) {
					tail = head;
					head++;
				}
			} else {
				head++;
			}
		}
		int which_bin = strtol(tail, &head, 10);
		head++;
		bin_count[which_bin]++;
		printf("abin %u count %u\n", which_bin, bin_count[which_bin]);
		int goods_count = 1;
		while (*head != '#' && *head != '\0') {
			if (*head == '\t') {
				goods_count++;
			}
			head++;
		}
		if (goods_count == 1) {
			have_singelton[which_bin] = 1;
		}
//		printf("Bin %d count %u\n",which_bin,bin_count[which_bin]);
	}
	free(line);

	return bin_count;
}

int compare_int(const void* p1, const void* p2) {
//	struct bid b1 =
	float i1 = ((struct bid*) p1)->average;
	float i2 = ((struct bid*) p2)->average;
	////assert(((struct bid*)p1)->bin == ((struct bid*)p2)->bin);
	return i1 > i2 ? -1 : i1 < i2 ? 1 : 0;

}

struct bid * remove_from_list(struct bid * curr, struct bid * root) {

	if (curr->prev) { // if current node is not the first in the list
		curr->prev->next = curr->next; // then point prev to the next in the list
	} else {
		root = curr->next;
	}
	if (curr->next) { //if current node is not last in the list
		curr->next->prev = curr->prev; // then point the next node to the prev
		////assert(curr != curr->next->prev);
	}

	return root;
}

int get_next_best_good(struct configuration * conf, struct bid * curr) {
	int x;
	double total_goods_count[conf->goods];
	unsigned int numbids_count[conf->goods];
	for (x = 0; x < conf->goods; x++) {
		total_goods_count[x] = numbids_count[x] = 0;
	}

	while (curr) {
		int goods_count = 0;
		for (x = 0; x < SIZE; x++) {
			goods_count += __builtin_popcount((unsigned int) curr->alloc.a[x]);
		}
		for (x = 0; x < conf->goods; x++) {
			int word_index = x / WORDSIZE;
			int bit_index = x % WORDSIZE;
			int result = !!(curr->alloc.a[word_index] & (1 << bit_index));
			if (result)
				total_goods_count[x] += 3;
			numbids_count[x] += result;

		}
		curr = curr->next;
	}
	int min_pos = 0;
	float min = FLT_MIN;
	for (x = 0; x < conf->goods; x++) {
		float score = 0.0f;
		if (numbids_count[x]) {
			double avg = ((double) total_goods_count[x]); // / ((double) numbids_count[x]);
			score = ((double) numbids_count[x]) / avg;
			score = numbids_count[x];
			//printf("x %d score %.3f\n",x,score);
			if (score > min) {
				min = score;
				min_pos = x;
			}
		}
	}

	return min_pos;
}

void allocate_all_bids(FILE * fp, struct configuration * conf, unsigned int * have_singelton, unsigned int * bin_count) {
	conf->allocation = (struct allocation *) malloc(sizeof(struct allocation) * conf->bids);

	conf->id = (unsigned int *) malloc(sizeof(unsigned int) * conf->bids);
	conf->dummies = (unsigned int *) malloc(sizeof(unsigned int) * conf->bids);
	conf->bin = (unsigned char *) malloc(sizeof(unsigned int) * conf->bids);
	conf->offer = (unsigned int *) malloc(sizeof(float) * conf->bids);
	conf->average = (float *) malloc(sizeof(float) * conf->bids);
	conf->max_offset = (unsigned short *) malloc(sizeof(unsigned int) * conf->goods);

	conf->bin_count = bin_count;

	char * head = NULL;
	char * tail = NULL;
	char *line = NULL;
	unsigned long total_goods_count[conf->goods];

	unsigned int numbids_count[conf->goods];
	unsigned int goods[conf->goods];
	size_t len = 0;
	int x;
	unsigned int bin_index[conf->goods];
	bin_index[0] = 0;
	total_goods_count[0] = numbids_count[0] = 0;
	for (x = 1; x < conf->goods; x++) {
		bin_index[x] = bin_count[x - 1] + bin_index[x - 1];
		total_goods_count[x] = numbids_count[x] = 0;
	}
	struct bid * tmp_bids = (struct bid *) malloc(sizeof(struct bid) * conf->bids);
	struct bid * root = &tmp_bids[0]; //malloc(sizeof(struct bid));
	struct bid * curr = root;
	curr->next = NULL;
	curr->prev = NULL;
	for (x = 1; x < conf->bids; x++) {
		curr->next = &tmp_bids[x]; //malloc(sizeof(struct bid));
		curr->next->prev = curr;
		curr = curr->next;
		curr->next = NULL;
	}

	curr = root;
	while ((getline(&line, &len, fp)) != -1) {
		if (!isdigit(line[0])) {
			continue;
		}
		head = tail = line;

		while (*head != '\t' && *head != '\0') {
			head++;
		}
		int id = strtol(tail, &head, 10);
		tail = head;
		head++;
		//get offer or value
		while (*head != '\t' && *head != '\0') {
			head++;
		}
		unsigned int offer = strtol(tail, &head, 10);
		tail = head;
		head++;
		unsigned int goods_count = 0;
		unsigned int good = 0;
		unsigned int dummy = 0;
		unsigned int tmp_allocation[SIZE];
		for (x = 0; x < SIZE; x++) {
			tmp_allocation[x] = 0;
		}
		//reset the temporary goods array, used to determin the score
		goods_count = 0;
		for (x = 0; x < conf->goods; x++) {
			goods[x] = 0;
		}

		while (*head != '#' && *head != '\0') {
			if (*head == '\t') {
				good = strtol(tail, &head, 10);

				//sscanf(tail,"\t%u\t",tmp2);
				if (good < conf->goods) {
					tmp_allocation[(good / WORDSIZE)] += (1 << good);
					goods[goods_count] = good;
				} else {
					dummy = good;
				}
				tail = head;
				goods_count++;
			}
			head++;

		}
		if (dummy > 0) {
			goods_count--;
		}
		curr->average = (float) offer / (goods_count);
		for (x = 0; x < goods_count; x++) {
			total_goods_count[goods[x]] += goods_count;
			numbids_count[goods[x]]++;
		}
		curr->offer = offer;
		curr->bin = goods[0];
		curr->dummy = dummy;
		curr->id = id;
		for (x = 0; x < SIZE; x++) {
			curr->alloc.a[x] = tmp_allocation[x];
		}
		curr = curr->next;
//		printf("id %d bin %u count %u value %.3lf\n",bid_count,bin_for_bid,tmp_count[bin_for_bid],0);
		bin_index[goods[0]]++;
		//printf("hello\n");
	}
	free(line);
	float min = FLT_MAX;
	int min_pos = 0;
	int singleton_count = conf->bids - conf->singletons;
	for (x = 0; x < conf->goods; x++) {
		if (!have_singelton[x]) {
			int y;
			for (y = 1; y < SIZE; y++) {
				curr->alloc.a[x] = 0;
			}
			int word_index = x / WORDSIZE;
			int bit_index = x % WORDSIZE;
			curr->alloc.a[word_index] = (1 << bit_index);
			curr->offer = curr->average = 0.0f;
			curr->dummy = 0;
			curr->bin = x;
			curr->id = singleton_count;

			total_goods_count[x] += 1; //add one more to the score stat
			numbids_count[x] += 1; // also add one more to the number of bids to the score stat
			singleton_count++; // next singleton bid will have an consecutive bid id
			curr = curr->next;
		}
		double score = 0;
		double avg;
		if (numbids_count[x]) {
			printf("x %d total good count %lu, numbids_count %d\n", x, total_goods_count[x], numbids_count[x]);
			avg = ((double) total_goods_count[x]) / ((double) numbids_count[x]);
			score = ((double) numbids_count[x]) / avg;
		}
		if (score < min) {
			min = score;
			min_pos = x;
		}
	}
	unsigned int bid_to_bit[conf->goods];

	for (x = 0; x < conf->goods; x++) {
		bid_to_bit[x] = 0;
	}
	printf("min %.3f pos %d\n", min, min_pos);
	int bid_bit_count = -1;
	struct bid * new_root = NULL;
	struct bid * new_curr = NULL;
	int bid_count = 0;
	while (root) {
		curr = root;
		min_pos = get_next_best_good(conf, curr);
		bid_count = 0;
		bid_bit_count++;
		bid_to_bit[min_pos] = bid_bit_count;
		while (curr) {

			int word_index = min_pos / WORDSIZE;
			int bit_index = min_pos % WORDSIZE;
			struct bid * next = curr->next;
			if (curr->alloc.a[word_index] & (1 << bit_index)) {
				curr->bin = bid_bit_count;
				if (!new_root) {
					root = remove_from_list(curr, root);
					new_root = curr;
					//curr = curr->next;
					new_curr = new_root;
					new_curr->prev = NULL;

				} else {
					root = remove_from_list(curr, root);
					new_curr->next = curr;
					new_curr->next->prev = new_curr;
					//curr = curr->next;

					new_curr = new_curr->next;
				}
				new_curr->next = NULL;
				bid_count++;
			}
			curr = next;

		}
		curr = root;
		conf->bin_count[bid_bit_count] = bid_count;
		conf->max_offset[bid_bit_count] = bid_count - 1;
		min_pos = get_next_best_good(conf, curr);
		printf("min pos %u\n", min_pos);
	}
	new_curr = new_root;
	while (new_curr) {
		struct allocation tmp;
		for (x = 0; x < SIZE; x++) {
			tmp.a[x] = 0;
		}
		for (x = 0; x < conf->goods; x++) {
			int bit_index = x % WORDSIZE;
			int word_index = x / WORDSIZE;
			if ((new_curr->alloc.a[word_index] & (1 << bit_index))) {
				tmp.a[word_index] |= (1 << bid_to_bit[x]);
//				printf("good %d translation %d\n",x,bid_to_bit[x]);
			}
		}
		for (x = 0; x < SIZE; x++) {
			new_curr->alloc.a[x] = tmp.a[x];
		}
		/* printf("id %u res %u\n",new_curr->id,new_curr->alloc.a[0]); */
		/* exit(0); */
		/* printf("%u bin %u\n",new_curr->id,bit_to_bid[new_curr->bin]); */
		new_curr = new_curr->next;
	}
	printf("total bids %u\n", conf->bids);

	for (x = 0; x < conf->goods; x++) {
		printf("%d %u\n", x, conf->bin_count[x]);
	}

//	exit(0);
	int y;

	int bin_index2[conf->goods];
	bin_index2[0] = 0;

	for (x = 1; x < conf->goods; x++) {
		bin_index2[x] = bin_count[x - 1] + bin_index2[x - 1];
	}

	x = 0;

	struct bid * lhead, *ltail;
	ltail = new_root;
	while (ltail) {
		int good = ltail->bin;
		lhead = ltail->next;
		while (lhead && lhead->bin == good) {
			if (lhead->average > ltail->average) {
				if (lhead->prev == ltail) {

					if (ltail->prev)
						ltail->prev->next = lhead;
					else
						new_root = lhead;
					if (lhead->next)
						lhead->next->prev = ltail;
					lhead->prev = ltail->prev;
					ltail->next = lhead->next;
					lhead->next = ltail;
					ltail->prev = lhead;
					struct bid * tmp = lhead;
					lhead = ltail;
					ltail = tmp;
				} else {

					struct bid *ltailprev, *ltailnext;
					////assert(ltail->next != lhead);
					////assert(lhead->prev != ltail);
					////assert(lhead->next != ltail);
					////assert(ltail->prev != lhead);
					ltailnext = ltail->next;
					ltailprev = ltail->prev;
					if (ltail->prev)
						ltail->prev->next = lhead;
					else
						new_root = lhead;
					if (ltail->next)
						ltail->next->prev = lhead;
					if (lhead->prev)
						lhead->prev->next = ltail;
					if (lhead->next)
						lhead->next->prev = ltail;
					ltail->next = lhead->next;
					ltail->prev = lhead->prev;
					lhead->next = ltailnext;
					lhead->prev = ltailprev;
					struct bid * tmp = lhead;
					lhead = ltail;
					ltail = tmp;
				}
			}
			lhead = lhead->next;
		}
		ltail = ltail->next;
	}
	new_curr = new_root;

	while (new_curr) {

		int index = x;
		for (y = 0; y < SIZE; y++) {
			conf->allocation[index].a[y] = new_curr->alloc.a[y];
			////assert(conf->allocation[index].a[y] == new_curr->alloc.a[y]);
		}
		conf->bin[index] = new_curr->bin;
		////assert(
		//	conf->allocation[index].a[conf->bin[index]/WORDSIZE] & (1<< conf->bin[index]));
		////assert(x >= bin_index2[conf->bin[index]]);
		////assert(x < bin_count[conf->bin[index]]+bin_index2[conf->bin[index]]);

		conf->offer[index] = new_curr->offer;
		conf->dummies[index] = new_curr->dummy;
		conf->id[index] = new_curr->id;
		conf->average[index] = new_curr->average;
		new_curr = new_curr->next;
		x++;
	}
	free(tmp_bids);

	return;
}

#define DEBUG (0)
#define THREADS (1024)
#define WARPS (THREADS/32)
#define GOODS (32)

#define HELLO(X) {	if (laneid == 0) { \
printf("hello %d, %u\n", __LINE__,X);\
}\
}

//__constant__ unsigned int max_index[GOODS+1];
//__constant__ unsigned int bin_index[GOODS+1];
#define ALLOC_32BIT_WORDS (1)
#define PUT_I (0)
#define ALLOC_I (1)
#define FLUSH (2)
#define FLUSH_RESET (32)
#define WARP_SIZE (32)

#define SHARED_ENTRIES (1000)

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif
static __device__ __inline__ unsigned int __load_noncache(unsigned int *ptr) {
	unsigned int ret;
	asm volatile ("ld.global.lu.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr));
	return ret;
}

static __device__ __inline__ unsigned int __load_supercache(unsigned int *ptr) {
	unsigned int ret;
	asm volatile ("ld.global.ca.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr));
	return ret;
}
static __device__ __inline__ void __prefetch(unsigned int * ptr) {
	asm volatile ("prefetch.global.L2 [%0];" :: "l"(ptr));
}


//extern __shared__ unsigned int array[];

__global__ void merge_dominate_prune(struct allocation * in1, unsigned int * val1, unsigned int count1,
									struct allocation * in2, unsigned int * val2, unsigned int count2,
									struct allocation * out, unsigned int * outval,
									unsigned int * atom_count, unsigned int shared_indexes, unsigned int * block_aloc) {

	__shared__ struct allocation  new_alloc[SHARED_ENTRIES];// = (struct allocation *) array; // shared mem
	__shared__ unsigned int new_value[SHARED_ENTRIES];// = (unsigned int *) &array[shared_indexes * ALLOC_32BIT_WORDS];// shared mem

	__shared__ int shared_atom[3];
	if (threadIdx.x == 0) {
		shared_atom[0] = shared_atom[1] = 0;
		shared_atom[FLUSH] = FLUSH_RESET;
	}
	__syncthreads();
	struct allocation my_alloc;
	const char laneid = threadIdx.x % 32;
	const unsigned int ltmask = (1 << laneid) - 1;
	unsigned int my_val;

	int allocate = 1;
	int index = 0;

	while (allocate) {
		if(laneid < count2) {
			__prefetch(&in2[laneid].a[0]);
			__prefetch(&val2[laneid]);
		}
		if (laneid == 0) {
//			index = atomicAdd(&shared_atom[ALLOC_I], WARP_SIZE);
			index = atomicAdd(block_aloc, WARP_SIZE);
		}

		index = __shfl(index, 0);
		if(index >= count1) {
			allocate = 0;
			break;
		}
		int t_index = index + laneid;
		if (t_index < count1) {
			my_alloc.a[0] = __load_noncache(&in1[t_index].a[0]);
			my_val = __load_noncache(&val1[t_index]);
		} else {
			my_alloc.a[0] = UINT_MAX;
			my_val = 0;
		}

		for (int x = 0; x < count2; x +=32) {
			struct allocation tmp;
			unsigned int t_val = 0;

			if (laneid + x < count2) {
				tmp.a[0] = __load_supercache(&in2[laneid + x].a[0]);
				t_val = __load_supercache(&val2[laneid + x]);
			} else {
				tmp.a[0] = UINT_MAX;
			}

			unsigned int available = __ballot((laneid + x) < count2);
			if (!available) {
				//allocate = 0;
				continue;
			}
#pragma unroll 32
			for (int src_lane = 0; (available); src_lane+=2, available >>=2) {

				struct allocation put;//my_alloc;
				struct allocation put2;//my_alloc;
				put.a[0] = 0;
				put2.a[0] = 0;
				unsigned int get1 = __shfl((int) tmp.a[0], src_lane);
				unsigned int get2 = __shfl((int) tmp.a[0], src_lane+1);

				if ((my_alloc.a[0] & get1) == 0) {
					put.a[0] = get1 | my_alloc.a[0];
				}

				if ((my_alloc.a[0] & get2) == 0) {
					put2.a[0] = get2 | my_alloc.a[0];
				}

				unsigned int status1 = __ballot(put.a[0]);
				unsigned int status2 = __ballot(put2.a[0]);

				if (!status1 && !status2) {
					continue;
				}
				//get value from src_lane
				unsigned int p_val = my_val + ((unsigned int) __shfl((int) t_val, src_lane));
				unsigned int p_val2 = my_val + ((unsigned int) __shfl((int) t_val, src_lane+1));
				int put_index;

				if (laneid == 0) {
					//laneid 0 gets the index in shared memory
					if (shared_atom[PUT_I] < shared_indexes) {
						put_index = atomicAdd(&shared_atom[PUT_I], __popc(status1)+__popc(status2));
					} else {
						put_index = shared_indexes;
					}

				}

				put_index = __shfl(put_index, 0);
				if (put.a[0] || put2.a[0]) {
					put_index += __popc(status1 & ltmask) + __popc(status2 & ltmask);

					if (put_index < shared_indexes) {
						if (put.a[0]) {
							new_alloc[put_index] = put;
							new_value[put_index] = p_val;

						}
						if (put2.a[0] && put_index + 1 < shared_indexes) {
							new_alloc[put_index + 1] = put2;
							new_value[put_index + 1] = p_val2;
							put2.a[0] = 0;
						}
						//make sure that it does not also writes to global memory if one or more threads in the warp
						//is outside the bounds
						put.a[0] = 0;
					}

				}

				status1 = __ballot((put_index >= shared_indexes) && put.a[0]);
				status2 = __ballot((put_index+1 >= shared_indexes) && put2.a[0]);
				//if overflow the shared memory resort to global
				//could do it with batches if we reserve shared memory, and when found 32 compatible bids
				//we write to global memory but that is subjected to fragmentation



				if (status1) {
					int left = 0;
					if (laneid == 0) {
						left = __popc(status1);
						put_index = atomicAdd(atom_count, left);

						atomicCAS(&shared_atom[FLUSH], FLUSH_RESET, threadIdx.x / 32);

					}
					put_index = __shfl(put_index, 0);
					if (put.a[0]) {
						put_index += __popc(status1 & ltmask);
						out[put_index] = put;
						outval[put_index] = p_val;
					}
				}

				if (status2) {
					int left = 0;
					if (laneid == 0) {
						left = __popc(status2);
						put_index = atomicAdd(atom_count, left);

						atomicCAS(&shared_atom[FLUSH], FLUSH_RESET, threadIdx.x / 32);

					}
					put_index = __shfl(put_index, 0);
					if (put2.a[0]) {
						put_index += __popc(status2 & ltmask);
						out[put_index] = put2;
						outval[put_index] = p_val2;
					}
				}

				//flush shared memory to global memory
				if (shared_atom[FLUSH] == (threadIdx.x / 32)) {

					if (laneid == 0) {
						put_index = atomicAdd(atom_count, shared_indexes);
					}
					put_index = __shfl(put_index, 0);
					for (int outi = laneid; outi < shared_indexes; outi += 32) {
						out[outi + put_index] = new_alloc[outi];
						outval[outi + put_index] = new_value[outi];
					}
					shared_atom[PUT_I] = 0;
					shared_atom[FLUSH] = FLUSH_RESET;
				}

			}

		}

		if (allocate == 0) {
			break;
		}

	}
	__syncthreads();
	int put_index = *atom_count;
	int max_index = shared_atom[PUT_I];
	for(int outi = threadIdx.x; outi < max_index; outi+= blockDim.x) {
		out[outi + put_index] = new_alloc[outi];
		outval[outi + put_index] = new_value[outi];
	}
}

void test_run(void) {


	struct allocation * in1; unsigned int * val1;
	struct allocation * in2; unsigned int * val2;
	struct allocation * out; unsigned int * outval;

	const int bids1 = 600000;
	const int bids2 = 600000;

	struct allocation allocs1[bids1];
	struct allocation allocs2[bids2];
	unsigned int values[bids1];
	unsigned int max_bids = 100;//((bids1+1)*(bids2+1)-1)*0.5;
	unsigned int * rolling_index;
	printf("Setting up memory, max bids %u, tot amount %u\n",max_bids,max_bids*2+bids1*4);
	CUDA_CHECK_RETURN(cudaDeviceReset());
	CUDA_CHECK_RETURN(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	int _max = 0;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &rolling_index, sizeof(unsigned int)*2));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &val1, sizeof(unsigned int)*bids1));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &val2, sizeof(unsigned int)*bids2));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &outval, sizeof(unsigned int)*max_bids));


	CUDA_CHECK_RETURN(cudaMalloc((void**) &in1, sizeof(struct allocation)*bids1));


	CUDA_CHECK_RETURN(cudaMalloc((void**) &in2, sizeof(struct allocation)*bids2));


	CUDA_CHECK_RETURN(cudaMalloc((void**) &out, sizeof(struct allocation)*max_bids));


	CUDA_CHECK_RETURN( cudaMemset(rolling_index, 0, sizeof(unsigned int)*2));

	int x;
	for(x=0;x<bids1;x++) {
		allocs1[x].a[0] = 1;
		values[x] = 1;
	}
	for(x=0;x<bids2;x++) {
		allocs2[x].a[0] = 1;
	}
	CUDA_CHECK_RETURN( cudaMemcpy(rolling_index, &_max, sizeof(unsigned int)*1, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(in1, &allocs1, sizeof(struct allocation)*bids1, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(in2, &allocs2, sizeof(struct allocation)*bids2, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(val1, &values, sizeof(unsigned int)*bids1, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(val2, &values, sizeof(unsigned int)*bids2, cudaMemcpyHostToDevice));

//	CUDA_CHECK_RETURN( cudaMemset(val1,1, sizeof(unsigned int)*bids));
//	CUDA_CHECK_RETURN( cudaMemset(val2,1, sizeof(unsigned int)*bids));


	int blocks = 10;
	cudaStream_t stream1;
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	unsigned int cache = (48*1024)-16;
	cache = (cache)/8;
	printf("Running kernel, size %lu size alloc %lu size uint %u\n",size_t(cache),sizeof(struct allocation),sizeof(unsigned int));
	time_t start = time(NULL);
	merge_dominate_prune <<<blocks, THREADS,0,stream1>>>(in1,val1,bids1,
																	   in2,val2,bids2,
																	   out,outval,
																	   rolling_index,SHARED_ENTRIES,rolling_index+1);


//	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	printf("time %lu \n",  time(NULL) - start);
// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
//	CUDA_CHECK_RETURN(
//		cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

//	for (i = 0; i < WORK_SIZE; i++)
//		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);

	CUDA_CHECK_RETURN( cudaMemcpy(&_max, rolling_index, sizeof(unsigned int)*1, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaDeviceReset());
	printf("max %u\n", _max);

}
/*
template<int uses_dummy, int single_word>
__launch_bounds__(THREADS,2)
__global__ void calc_best2(unsigned int * max, struct allocation * _allocation, unsigned int * _offer, unsigned short * _max_index, unsigned short * _bin_index,
		unsigned int * dummies, unsigned int * id, unsigned int bids, unsigned int * host_max, unsigned int * rolling_index) {

	extern __shared__ struct allocation allocation[];
	__shared__ struct allocation curr_allocation[WARPS];
	__shared__ unsigned int offer[132];
	__shared__ unsigned short max_index[GOODS];
	__shared__ unsigned short bin_index[GOODS];
	__shared__ unsigned int value[WARPS];
	__shared__ unsigned int shared_max;
	__shared__ unsigned short count[WARPS][GOODS];
	__shared__ unsigned char allocation_id_index[WARPS];
	__shared__ unsigned short allocation_id[WARPS][GOODS];

	const char laneid = threadIdx.x % 32;
	const char warpid = threadIdx.x / 32;
	int x;

	for (x = threadIdx.x; x < bids; x += blockDim.x) {
		allocation[x] = _allocation[x];
		offer[x] = _offer[x];
	}

	if (threadIdx.x < GOODS) {
		bin_index[threadIdx.x] = _bin_index[threadIdx.x];
		max_index[threadIdx.x] = _max_index[threadIdx.x];
	}
	__threadfence_block();
	__syncthreads();

	new_coalition: ;
	int w_rolling_index;
	if (laneid == 0) {
		w_rolling_index = atomicAdd(rolling_index, 1); //could do it just for the block and then push to threads
		curr_allocation[warpid].a[0] = 0;
		value[warpid] = 0;
		allocation_id_index[warpid] = 0;
	}
	w_rolling_index = __shfl(w_rolling_index, 0);

	int good = 0;
	int binindex;

	while (w_rolling_index >= max_index[good] + 1) {
		binindex = bin_index[good];
		if (laneid == 0) {
			assert((allocation[max_index[good] + binindex].a[0] & curr_allocation[warpid].a[0]) == 0);
			curr_allocation[warpid].a[0] |= allocation[max_index[good] + binindex].a[0];
			//value[warpid] += offer[max_index[good] + binindex];
//			assert((curr_allocation[warpid].a[0] & ((1 << good)-1)) == ((1 << good)-1)     );
//			assert((max_index[good] + binindex) < bids);
//			assert(value[warpid] == 0);
		}
		w_rolling_index -= max_index[good] + 1;
		good++;
		if (good >= GOODS) {
			return;
		}
	}

	binindex = bin_index[good];
	if (laneid == 0) {
		curr_allocation[warpid].a[0] |= allocation[w_rolling_index + binindex].a[0];
		value[warpid] += offer[w_rolling_index + binindex];
	}

	for (x = laneid; x < (GOODS); x += 32) {
		count[warpid][x] = 0;
	}

	//start add the second bid ---------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------------------------
	if (laneid == 0) {
		if (value[warpid] > shared_max) {
			if (atomicMax(max, value[warpid]) < value[warpid]) {

				atomicMax(host_max, value[warpid]);
			}
			atomicMax(&shared_max, value[warpid]);
		}
	}
//	goto new_coalition;

//end add the second bid -----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------
	int status;
	int index;
	int allocate = 1; //could remove and just use goto
	int dealloc = 0;
//int ccount = 0;
	while (allocate || dealloc) {

		while (allocate) {

			if (laneid == 0) {
				x = 0;
				status = 1;
				while (status && x < GOODS) {
					status = curr_allocation[warpid].a[0] & (1 << x);
					if (status) {
						x++;
					}
				}
				good = x;
			}
			good = __shfl(good, 0);
			//if there are no more goods to allocate, exit allocation loop
			if (good >= (GOODS)) {
				//printf("dealloc full good\n");
				allocate = 0;
				dealloc = 1;
				break;
			}

			//if(good >= (GOODS - 2))

			////assert((curr_allocation[warpid].a[0]	& (1 << good)) == 0);
			int max_offset = max_index[good];
			binindex = bin_index[good];

			index = 0;

			status = 0;
			while (status == 0) {

				index = count[warpid][good] + laneid;

				if (index <= max_offset) {
					int y;

					for (y = 0; y < SIZE; y++) {
						status |= allocation[index + binindex].a[y] & (curr_allocation[warpid].a[y]);
					}
					////assert(bin[index+binindex] == good);
					status = __ballot((status == 0));

					if (status == 0) { //if no thread have found a compatible bid
						if (laneid == 0) {
							count[warpid][good] += 32;
						}
					} else if (laneid == 0) {
						index = __ffs(status) - 1 + count[warpid][good];
						count[warpid][good] = index + 1;
					}

				}
				status = __shfl((status), 0);
				//if the count index is greater than the maximum offset
				if (count[warpid][good] > max_offset) {
					break;
				}

			}

			//status = __shfl(status, 0);

			//if count is greater than maximum offset and we did not find a suitable bid
			if (dealloc & !status) {
				dealloc = 1;
				if (laneid == 0) {
					count[warpid][good] = 0;
				}
				allocate = 0;
				break;
			}
			index = __shfl(index, 0);
			//parallel --------------------------------------------------------------------------------------------------
			// add the goods
			if (laneid < SIZE) {
				////assert((curr_allocation[warpid].a[laneid] &	allocation[index + binindex].a[laneid]) == 0);
				curr_allocation[warpid].a[laneid] |= allocation[index + binindex].a[laneid];
			}

			//dummy bid for the bid we allocation
			if (laneid == 0) {

				value[warpid] += offer[index + binindex];

				//which bid we allocated
				allocation_id[warpid][allocation_id_index[warpid]] = index + binindex;
				allocation_id_index[warpid]++;

				if (value[warpid] > shared_max) {
					//atomicMax(max, value[warpid]);
					atomicMax(&shared_max, value[warpid]);

					if (atomicMax(max, value[warpid]) < value[warpid]) {

						(*host_max = value[warpid]);
					}

				}

			}
			//continue;
			continue;
			float mySum = 0.0f;
			float tmpSum = 0.0f;
			unsigned int shift = (1 << laneid);
			int g_status = ((curr_allocation[warpid].a[0] & shift) == 0);
			max_offset = max_index[laneid];
			binindex = bin_index[laneid];
			index = 0;

			unsigned int recive_from = __ballot(g_status) & ((1 << (laneid) - 1)); // will give all the free goods
			while (g_status && __ballot(recive_from)) {
				//0011001
				//g_status &= __ballot(g_status) ^ shift;
				unsigned int alloc;
				status = 1;
				if (index <= max_offset) {
					alloc = allocation[index + binindex].a[0];
					status = curr_allocation[warpid].a[0] & alloc;
					if (status == 0) { //no collision
						tmpSum = offer[index + binindex] / 3;
						mySum = fmaxf(mySum, tmpSum);
					}
					index++;
				} else {
					alloc = 0;
				}

				unsigned int available = __ballot((status == 0));
				while (available) {

					unsigned int thread = __ffs(available) - 1;
					float shf_sum = __shfl(tmpSum, thread);
					unsigned int ret = __shfl((int) alloc, thread);

					if (shf_sum <= mySum || !ret) {
						recive_from ^= 1 << thread; //remove thread from good recipient
					} else if ((ret & shift)) {
						mySum = shf_sum;
						recive_from ^= 1 << thread; //remove thread from good recipient
					}
					available ^= 1 << thread;
				}
				g_status = __ballot((index <= max_offset));
			}

			mySum += __shfl_xor(mySum, 16);
			mySum += __shfl_xor(mySum, 8);
			mySum += __shfl_xor(mySum, 4);
			mySum += __shfl_xor(mySum, 2);
			mySum += __shfl_xor(mySum, 1);
			if ((value[warpid] + (unsigned int) mySum) <= shared_max) {
				//assert(mySum > 0.0f);
				dealloc = 1;
				allocate = 0;
			}

			//parallel --------------------------------------------------------------------------------------------------
		}

		while (dealloc) {
			dealloc = 0;
			allocate = 1;
			if (allocation_id_index[warpid] == 0) {
				dealloc = 0;
				allocate = 1;
				//		return;
				goto new_coalition;
			}
			if (laneid == 0) {
				allocation_id_index[warpid]--;
			}

			//printf("index %u\n",conf->allocation_id_index);
			int dealloc_index = allocation_id[warpid][allocation_id_index[warpid]];
			if (allocation_id_index[warpid] > GOODS) {
				value[warpid] += 1;
			}
			int dealloc_good; //bin[dealloc_index];
			if (laneid == 0) {
				//assert(offer[dealloc_index] <= value[warpid]);
				value[warpid] -= offer[dealloc_index];
				dealloc_good = __ffs((int) (allocation[dealloc_index].a[0])) - 1;
			}
			dealloc_good = __shfl(dealloc_good, 0);

			if (laneid < SIZE) {
				curr_allocation[warpid].a[laneid] ^= allocation[dealloc_index].a[laneid];
			}

			if (count[warpid][dealloc_good] > max_index[dealloc_good]) {
				//	printf("re-de-alloc good %u\n",dealloc_good);
				if (laneid == 0) {
					count[warpid][dealloc_good] = 0;
				}
				dealloc = 1;
				allocate = 0;
			}

			if (allocation_id_index[warpid] == 0) {
				dealloc = 0;
				allocate = 1;
				//		return;
				goto new_coalition;
			}

		}

	}

}

template<int uses_dummy, int single_word>
__launch_bounds__(THREADS,2)
__global__ void calc_best3(unsigned int * max, struct allocation * _allocation, unsigned int * _offer, unsigned short * _max_index, unsigned short * _bin_index,
		unsigned int * dummies, unsigned int * id, unsigned int bids, unsigned int * host_max, unsigned int * rolling_index) {

	extern __shared__ struct allocation allocation[];
	__shared__ struct allocation curr_allocation[WARPS];
	__shared__ unsigned int offer[532];
	__shared__ unsigned short max_index[GOODS];
	__shared__ unsigned short bin_index[GOODS];
	__shared__ unsigned short count[WARPS][GOODS];
	__shared__ unsigned int value[WARPS];
	__shared__ unsigned int shared_max;
	__shared__ unsigned char allocation_id_index[WARPS];
	__shared__ unsigned short allocation_id[WARPS][GOODS - 1];

	const char laneid = threadIdx.x % 32;
	const char warpid = threadIdx.x / 32;
	int x;

	for (x = threadIdx.x; x < bids; x += blockDim.x) {
		allocation[x] = _allocation[x];
		offer[x] = _offer[x];
	}

	if (threadIdx.x < GOODS) {
		bin_index[threadIdx.x] = _bin_index[threadIdx.x];
		max_index[threadIdx.x] = _max_index[threadIdx.x];
	}
	__threadfence_block();
	__syncthreads();

	new_coalition: ;
	int good = 0;
	int binindex;
	if (laneid == 0) {
		int w_rolling_index;
		w_rolling_index = atomicAdd(rolling_index, 1); //could do it just for the block and then push to threads
		curr_allocation[warpid].a[0] = 0;
		value[warpid] = 0;
		allocation_id_index[warpid] = 0;

		w_rolling_index = __shfl(w_rolling_index, 0);
		int should_return = 0;
		while (w_rolling_index >= max_index[good] + 1) {
			binindex = bin_index[good];
			assert((allocation[max_index[good] + binindex].a[0] & curr_allocation[warpid].a[0]) == 0);
			curr_allocation[warpid].a[0] |= allocation[max_index[good] + binindex].a[0];
			value[warpid] += offer[max_index[good] + binindex];
//			assert((curr_allocation[warpid].a[0] & ((1 << good)-1)) == ((1 << good)-1)     );
//			assert((max_index[good] + binindex) < bids);
//			assert(value[warpid] == 0);
			w_rolling_index -= max_index[good] + 1;
			good++;
			if (good >= GOODS || max_index[good] == 0) {
				//w_rolling_index + binindex
				curr_allocation[warpid].a[0] = 0;
				should_return = 1;
				break;
			}
		}
		if (!should_return) {
			binindex = bin_index[good];
			curr_allocation[warpid].a[0] |= allocation[w_rolling_index + binindex].a[0];
			value[warpid] += offer[w_rolling_index + binindex];
		}
	}
	if (curr_allocation[warpid].a[0] == 0) {
		return;
	}

	for (x = laneid; x < (GOODS); x += 32) {
		count[warpid][x] = 0;
	}

	//start add the second bid ---------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------------------------
	if (laneid == 0) {
		if (value[warpid] > shared_max) {
			if (atomicMax(max, value[warpid]) < value[warpid]) {

				atomicMax(host_max, value[warpid]);
			}
			atomicMax(&shared_max, value[warpid]);
		}
	}
//	goto new_coalition;

//end add the second bid -----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------
	unsigned int status;
	int index;
	int allocate = 1; //could remove and just use goto
	int dealloc = 0;
//int ccount = 0;
	while (allocate || dealloc) {

		while (allocate) {

			status = __ballot((curr_allocation[warpid].a[0] & (1 << laneid)) == 0);
			if (status == 0) {
				allocate = 0;
				dealloc = 1;
				break;
			}

			if (laneid == 0) {
				good = __ffs(status) - 1;
			}
			good = __shfl(good, 0);
			////assert((curr_allocation[warpid].a[0]	& (1 << good)) == 0);
			int max_offset = max_index[good];
			if (max_offset == 0) {
				allocate = 0;
				dealloc = 1;
				break;
			}
			binindex = bin_index[good];

			index = 0;

			status = 0;
			while (status == 0) {

				index = count[warpid][good] + laneid;
				//status = 1;
				if (index <= max_offset) {
					////assert(bin[index+binindex] == good);
					status = __ballot(((allocation[index + binindex].a[0] & curr_allocation[warpid].a[0]) == 0));

					if (status == 0) { //if no thread have found a compatible bid
						if (laneid == 0) {
							count[warpid][good] += 32;
						}
					} else if (laneid == 0) {
						index = __ffs(status) - 1 + count[warpid][good];
						count[warpid][good] = index + 1;
					}

				}
				status = __shfl((int) (status), 0);
				//if the count index is greater than the maximum offset
				if (count[warpid][good] > max_offset) {
					dealloc = 1;
					break;
				}

			}

			//status = __shfl(status, 0);

			//if count is greater than maximum offset and we did not find a suitable bid
			if (dealloc && (status == 0)) {
				//dealloc = 1;
				if (laneid == 0) {
					count[warpid][good] = 0;
				}
				allocate = 0;
				break;
			}
			index = __shfl(index, 0);
			//parallel --------------------------------------------------------------------------------------------------
			// add the goods
			if (laneid < SIZE) {
				////assert((curr_allocation[warpid].a[laneid] &	allocation[index + binindex].a[laneid]) == 0);
				curr_allocation[warpid].a[laneid] |= allocation[index + binindex].a[laneid];
			}

			//dummy bid for the bid we allocation
			if (laneid == 0) {

				value[warpid] += offer[index + binindex];

				//which bid we allocated
				allocation_id[warpid][allocation_id_index[warpid]] = index + binindex;
				allocation_id_index[warpid]++;

				if (value[warpid] > shared_max) {
					//atomicMax(max, value[warpid]);
					atomicMax(&shared_max, value[warpid]);

					if (atomicMax(max, value[warpid]) < value[warpid]) {

						(*host_max = value[warpid]);
					}

				}

			}
			//continue;
			continue;
			float mySum = 0.0f;
			float tmpSum = 0.0f;
			unsigned int shift = (1 << laneid);
			int g_status = ((curr_allocation[warpid].a[0] & shift) == 0);
			max_offset = max_index[laneid];
			binindex = bin_index[laneid];
			index = 0;

			unsigned int recive_from = __ballot(g_status) & ((1 << (laneid) - 1)); // will give all the free goods
			while (g_status && __ballot(recive_from)) {
				//0011001
				//g_status &= __ballot(g_status) ^ shift;
				unsigned int alloc;
				status = 1;
				if (index <= max_offset) {
					alloc = allocation[index + binindex].a[0];
					status = curr_allocation[warpid].a[0] & alloc;
					if (status == 0) { //no collision
						tmpSum = offer[index + binindex] / 3;
						mySum = fmaxf(mySum, tmpSum);
					}
					index++;
				} else {
					alloc = 0;
				}

				unsigned int available = __ballot((status == 0));
				while (available) {

					unsigned int thread = __ffs(available) - 1;
					float shf_sum = __shfl(tmpSum, thread);
					unsigned int ret = __shfl((int) alloc, thread);

					if (shf_sum <= mySum || !ret) {
						recive_from ^= 1 << thread; //remove thread from good recipient
					} else if ((ret & shift)) {
						mySum = shf_sum;
						recive_from ^= 1 << thread; //remove thread from good recipient
					}
					available ^= 1 << thread;
				}
				g_status = __ballot((index <= max_offset));
			}

			mySum += __shfl_xor(mySum, 16);
			mySum += __shfl_xor(mySum, 8);
			mySum += __shfl_xor(mySum, 4);
			mySum += __shfl_xor(mySum, 2);
			mySum += __shfl_xor(mySum, 1);
			if ((value[warpid] + (unsigned int) mySum) <= shared_max) {
				//assert(mySum > 0.0f);
				dealloc = 1;
				allocate = 0;
			}

			//parallel --------------------------------------------------------------------------------------------------
		}

		while (dealloc) {
			dealloc = 0;
			allocate = 1;
			if (allocation_id_index[warpid] == 0) {
				dealloc = 0;
				allocate = 1;
				//		return;
				goto new_coalition;
			}
			if (laneid == 0) {
				allocation_id_index[warpid]--;
			}

			//printf("index %u\n",conf->allocation_id_index);
			int dealloc_index = allocation_id[warpid][allocation_id_index[warpid]];
			if (allocation_id_index[warpid] > GOODS) {
				value[warpid] += 1;
			}
			int dealloc_good; //bin[dealloc_index];
			if (laneid == 0) {
				//assert(offer[dealloc_index] <= value[warpid]);
				value[warpid] -= offer[dealloc_index];
				dealloc_good = __ffs((int) (allocation[dealloc_index].a[0])) - 1;
			}
			dealloc_good = __shfl(dealloc_good, 0);

			if (laneid < SIZE) {
				curr_allocation[warpid].a[laneid] ^= allocation[dealloc_index].a[laneid];
			}

			if (count[warpid][dealloc_good] > max_index[dealloc_good]) {
				//	printf("re-de-alloc good %u\n",dealloc_good);
				if (laneid == 0) {
					count[warpid][dealloc_good] = 0;
				}
				dealloc = 1;
				allocate = 0;
			}

			if (allocation_id_index[warpid] == 0) {
				dealloc = 0;
				allocate = 1;
				//		return;
				goto new_coalition;
			}

		}

	}

}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int setup_mem_and_run(struct configuration * conf) {
	int i;

	struct configuration transfer;
	conf->bin_index = (unsigned short *) malloc(sizeof(short) * conf->goods);
	conf->bin_index[0] = 0;
	for (i = 1; i < conf->goods; i++) {
		conf->bin_index[i] = conf->bin_index[i - 1] + conf->bin_count[i - 1];
	}

	int bids = conf->bids;
	int goods = conf->goods;
	unsigned int * max;
	unsigned int _max = 0;
	unsigned int * max_host;

	unsigned int * rolling_index;
	printf("Setting up memory\n");
	CUDA_CHECK_RETURN(cudaDeviceReset());
	//CUDA_CHECK_RETURN(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &rolling_index, sizeof(unsigned int)));

	CUDA_CHECK_RETURN( cudaHostAlloc(&max_host,sizeof(unsigned int),cudaHostAllocPortable));
	*max_host = 0;
	CUDA_CHECK_RETURN( cudaMalloc((void**) &transfer.bin_index, sizeof(unsigned short)*goods));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &max, sizeof(unsigned int)));
	CUDA_CHECK_RETURN( cudaMalloc((void**) &transfer.allocation, sizeof(struct allocation) * bids));
//CUDA_CHECK_RETURN(cudaMalloc((void**) &transfer.bin, sizeof(char) * bids));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &transfer.id, sizeof(int) * bids));
	CUDA_CHECK_RETURN( cudaMalloc((void**) &transfer.dummies, sizeof(int) * bids));
	CUDA_CHECK_RETURN( cudaMalloc((void**) &transfer.offer, sizeof(float) * bids));
	CUDA_CHECK_RETURN( cudaMalloc((void**) &transfer.average, sizeof(float) * bids));
//	CUDA_CHECK_RETURN(
//			cudaMalloc((void**) &transfer.bin_count, sizeof(int) * goods));
	CUDA_CHECK_RETURN( cudaMalloc((void**) &transfer.max_offset, sizeof(short) * goods));
//CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE));
	printf("%d goods\n", goods);
	CUDA_CHECK_RETURN( cudaMemcpy(transfer.allocation, conf->allocation, sizeof(struct allocation) * bids, cudaMemcpyHostToDevice));

//	printf("size %lu \n",sizeof(bin_index));
//	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(bin_index,conf->bin_index,sizeof(bin_index)));
//CUDA_CHECK_RETURN(cudaMemcpyToSymbol(max_index,conf->max_offset,sizeof(max_index)));
//CUDA_CHECK_RETURN(
//		cudaMemcpy(transfer.bin, conf->bin, sizeof(char) * bids, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(transfer.id, conf->id, sizeof(int) * bids, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(transfer.dummies, conf->dummies, sizeof(int) * bids, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(transfer.offer, conf->offer, sizeof(unsigned int) * bids, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(transfer.average, conf->average, sizeof(float) * bids, cudaMemcpyHostToDevice));
//	CUDA_CHECK_RETURN(
//			cudaMemcpy(transfer.bin_count, conf->bin_count, sizeof(int) * goods, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(transfer.max_offset, conf->max_offset, sizeof(short) * goods, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(max, &_max, sizeof(unsigned int)*1, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN( cudaMemcpy(rolling_index, &_max, sizeof(unsigned int)*1, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN( cudaMemcpy(transfer.bin_index, conf->bin_index, sizeof(short) * goods, cudaMemcpyHostToDevice));

//CUDA_CHECK_RETURN(cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));
	int blocks = 10;
	cudaStream_t stream1;
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));

	printf("Running kernel\n");
	//calc_best3<0, 1> <<<blocks, THREADS,sizeof(struct allocation)*conf->bids,stream1>>>(max,
		//	transfer.allocation, transfer.offer,transfer.max_offset,transfer.bin_index, transfer.dummies,
			//transfer.id, conf->bids,max_host,rolling_index);
	unsigned int tmp_max = 0;
	time_t start = time(NULL);
	while (cudaStreamQuery(stream1) == cudaErrorNotReady) {
		sleep(1);
		unsigned int t = *max_host;
		if (tmp_max < t) {
			tmp_max = t;
			printf("New max %u time %ull\n", tmp_max, time(NULL) - start);
		}

	}
//	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
//	CUDA_CHECK_RETURN(
//		cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

//	for (i = 0; i < WORK_SIZE; i++)
//		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);

	CUDA_CHECK_RETURN( cudaMemcpy(&_max, max, sizeof(unsigned int)*1, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaDeviceReset());
	printf("max %u\n", _max);
	return 0;
}

int main(int argc, char *argv[]) {
	test_run();
	return 0;
	int x;
	FILE * fp;
	fp = fopen(argv[1], "r");
	if (fp == NULL) {
		printf("Could not open file\n");
		exit(EXIT_FAILURE);
	}
	printf("hello\n");
	struct configuration * conf = get_configuration(fp);
	printf("goods %d\n", conf->goods);
	conf->singletons = 0;
	unsigned int * have_singleton = (unsigned int *) malloc(sizeof(int) * conf->goods);
	for (x = 0; x < conf->goods; x++) {
		have_singleton[x] = 0;
	}
	printf("goods %d\n", conf->goods);
	unsigned int * bin_count = get_bincount(fp, conf, have_singleton);
	printf("goods %d\n", conf->goods);
	for (x = 0; x < conf->goods; x++) {
		if (!have_singleton[x]) {
			conf->singletons++;
			conf->bids++;
			bin_count[x]++;
		}
		printf("bin %d count %u\n", x, bin_count[x]);
	}

	fclose(fp);
	fp = fopen(argv[1], "r");
	if (fp == NULL) {
		printf("Could not open file\n");
		exit(EXIT_FAILURE);
	}
	allocate_all_bids(fp, conf, have_singleton, bin_count);
	fclose(fp);
	for (x = 0; x < conf->bids; x++) {
		printf("x %d id %u, offer %u, bin %u, alloc %u, bin_count %u\n", x, conf->id[x], conf->offer[x], conf->bin[x], conf->allocation[x].a[0],
				conf->bin_count[conf->bin[x]]);

	}
	printf("words %u wordsize %lu\n", SIZE, WORDSIZE);
	free(have_singleton);
	setup_mem_and_run(conf);
//calc_best2(conf);
	free(conf->allocation);
	free(conf->bin);
	free(conf->id);
	free(conf->dummies);
	free(conf->bin_count);
	free(conf->max_offset);
	free(conf->offer);
	free(conf->average);
	free(conf->allocation_id);
	free(conf->allocation_dummy);
	free(conf);
	printf("Bye\n");
	exit(EXIT_SUCCESS);
}
