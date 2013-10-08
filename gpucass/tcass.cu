 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>

#include <sys/types.h>
#include <errno.h>
#include <ctype.h>

#if defined(__APPLE__)
#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/clock.h>
#else
#include <time.h>
#endif

#include "cuda_runtime_api.h"
#include "device_functions.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

enum symtype {EMPTY,INT,FLOAT,GOODS, BIDS, DUMMY, HASH, ERROR};
typedef struct {
    enum symtype utype;
    union {
        int ival;
        double fval;
    } u;
} symbol;

typedef struct {
    int goods;  //number of goods
    int bids;   //number of bids
    int dummy;  //number of dummy goods
    int *data;  //sctucture with the data
    int regsize; //size in int words of every register
    int words;  //number of int wods used to represent the bids
	int *bins;
	int *binelements;
	double *binsBests;
	double bestValue;
	
} problem_def;

typedef struct binstr_def {
	int binid;
	double dfactor;
	struct binstr_def *prev;
	struct binstr_def *next;
	int elements;
	int *ptr;
	int artificial;	
	double ratio;
} binstr;

double sampleTime() {
    // Time in nanosec
    struct timespec ts;
#if defined(__APPLE__)
	clock_serv_t cclock;
	mach_timespec_t mts;
	host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
	clock_get_time(cclock, &mts);
	mach_port_deallocate(mach_task_self(), cclock);
	ts.tv_sec = mts.tv_sec;
	ts.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return((ts.tv_sec+ts.tv_nsec/1000000000.0));

}
inline int    *bidIdentifier(problem_def* problem,int reg) {
    return (int *) &(problem->data)[(reg+1)*(problem->regsize)-1]  ;
}
inline int *bidValue(problem_def* problem,int reg) {
    return (int *) &(problem->data)[reg*(problem->regsize)];
}
inline unsigned int  *bidGoods(problem_def* problem,int reg) {
    return (unsigned int *) &(problem->data)[reg*(problem->regsize)+1]  ;
}
/*inline int    *bidDummyGood(problem_def* problem,int reg) {
    return (int *) &(problem->data)[(reg+1)*(problem->regsize)-2]  ;
}*/
inline int    bidSize(problem_def* problem, unsigned int* goods) {
    int i,num=0;
    for (i=0; i<problem->words; i++) {
        num+=__builtin_popcount(goods[i]);
    }
    return num;
}
inline void   setBitOn(unsigned int *bidset,int len,int bit) {
    int index=len- (bit/(sizeof(int)<<3))-1;
    bidset[index]=bidset[index] | (1<<((bit % (sizeof(int)<<3))));
}
inline int    checkBitOn(unsigned int *bidset,int len,int bit) {
    int index=len- (bit/(sizeof(int)<<3))-1;
    return bidset[index] & (1<<((bit % (sizeof(int)<<3))));
}
inline int    conflictBids(problem_def *problem, unsigned int *a,unsigned int *b) {
	int i;
	for (i=0; i<problem->words; i++) {
        if (a[i] & b[i]) return 1;
    }
	return 0;
}

void printHelp(char **str) {
    printf("Invalid options, valid usage\n");
    printf("%s -finputfile | -h \n",*str);
    printf("\nOptions\n");
    printf("-h       : This help\n");
    printf("-i <file>: input file\n");
    printf("-oi      : prices ar intgers\n");
    exit(0);
}

void printBidItemList(int *list, int regsize, int words,int n) {
	
	int *val = (int * ) &list[regsize*n];
	//if (*val<0) return;
	int *reg = &list[regsize*n+1];
	int i,j;
    printf(" %5i  ", *val);

    int good=0;
    for (i=words-1; i>=0; i--) {
        for (j=0; j<(sizeof(int)<<3); j++) {
            if ((reg[i]) & (1<<j)) printf(" %i ",good);
            good++;
        }
    }
    printf("\n");
}

void printBid(problem_def *problem, int n) {
    printf(" %i ", *bidIdentifier(problem,n));
    printf(" %i ", *bidValue(problem,n));
    int i,j;
    int good=0;
    for (i=problem->words-1; i>=0; i--) {
        for (j=0; j<(sizeof(int)<<3); j++) {
            if ((bidGoods(problem,n)[i]) & (1<<j)) printf(" %i ",good);
            good++;
        }
    }
  //  if (*bidDummyGood(problem,n)) printf(" %i* ", *bidDummyGood(problem,n));
    printf(" #  \n");
}
void printBid2(problem_def *problem, int n) {
    printf("Bid#: %i, ", *bidIdentifier(problem,n));
    printf("Amount: %i, Vec: ", (int) *bidValue(problem,n));
    int i;
   
	for (i=0;i<problem->goods;i++) {
		if (checkBitOn(bidGoods(problem,n),problem->words,i)) printf("1");
		else printf("0");
	}
	//	for (i=problem->goods;i<problem->dummy+(problem->goods);i++)	
	//		if (*bidDummyGood(problem,n)==i) printf("1"); 
	//		else printf("0");  

    printf("\n");
}

void printSolution(int *sol, int size, problem_def*p) {
    int i;
    printf("bids: ");
    for (i=0; i<size; i++) {
        printf(" %i ", *bidIdentifier(p,sol[i]));
    }
    printf("\n");
}
int checkParameters(int argc, char *argv[], char* filenameR, int * intprices) {
    int i,ret=0;
	*intprices=0;
	int opti=0;
	for (i=1; i<argc; i++) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
            case 'i':
                opti=1;
                FILE *fr;
                strcpy(filenameR,&argv[i+1][0]);
                fr=fopen(filenameR,"r");
				i++;
                if (!fr) {
                    printf("File not found\n");
                    return 0;
                } else {
                    fclose(fr);
                    ret=1;
                }
                break;
            case 'h' :
             
				break;
			case 'o' : 
	//			opto=1;
				if (argv[i][2]=='i') *intprices=1;
            default:
                break;
            }
        } else {
            printf("invalid option: %s\n",argv[i]);
            ret=0;
        }
    }
    if (opti==0) {
        printHelp(argv);
    }
    return ret;
}
int getSymbol(char *line,int *pos, symbol *ret) {
    int start;
    while ((line[*pos] != '\n') && (line[*pos] != '%')  && (line[*pos] != '\0')) {

        if (isdigit(line[*pos])) {
            start=*pos;
            ret->utype=INT;
            while (isdigit(line[*pos]) || (line[*pos]=='.')) {
                *pos=(*pos)+1;
                if (line[*pos]=='.') ret->utype=FLOAT;
            }
            if (ret->utype==INT  ) 	ret->u.ival=atoi(&line[start]);
            if (ret->utype==FLOAT) 	{
				ret->u.fval=atof(&line[start]);
			}
			return 0;
        }
        if (line[*pos]=='#') {
            *pos=(*pos)+1;
            ret->utype=HASH;
            return 0;
        }
        if (isalpha(line[*pos])) {
            if (strncmp(&line[*pos],"goods",5)==0) {
                ret->utype=GOODS;
                (*pos)=(*pos)+5;
            }
            else if (strncmp(&line[*pos],"bids",4)==0) {
                ret->utype=BIDS;
                (*pos)=(*pos)+4;
            }
            else if (strncmp(&line[*pos],"dummy",5)==0) {
                ret->utype=DUMMY;
                (*pos)=(*pos)+5;
            }
            else {
                ret->utype=ERROR;
                return 1;
            }
            return 0;
        }
        *pos=(*pos)+1;
    }
    ret->utype =EMPTY;
    return 0;
}
int readFile(problem_def *problem, char * filenameR, int intprices) {
    printf("Reading data from %s \n",filenameR);
    FILE *fr;
    char line[500],*tmp;
    int pos,count;
    int linenumber=0;
    int gbd=0;
    fr=fopen(filenameR,"r");
    int goods,bids,dummy;
    goods=bids=dummy=0;
    symbol a;
    while (!feof(fr)) {
        
        tmp=fgets(line,500,fr);
	//if (feof(fr)) break;
        if (tmp==NULL) printf("Error parsing file line %i\n",linenumber);
	linenumber++;        
	pos=0;

        if (gbd!=7) {
            do {
                getSymbol(line,&pos,&a);
                if (a.utype==ERROR) return linenumber;
                switch (a.utype) {
                case GOODS :
                    getSymbol(line,&pos,&a);
                    if (a.utype!=INT) return linenumber;
                    else goods=a.u.ival;
		    gbd = gbd | 1;	
                    break;
                case BIDS  :
                    getSymbol(line,&pos,&a);
                    if (a.utype!=INT) return linenumber;
                    else bids=a.u.ival;
		    gbd = gbd | 2;
                    break;
                case DUMMY :
                    getSymbol(line,&pos,&a);
                    if (a.utype!=INT) return linenumber;
                    else dummy=a.u.ival;
		    gbd = gbd | 4;
                    break;
                }
            } while (a.utype != EMPTY);

        } else break;

    }
    if (gbd!=7) {
        printf("No values for goods, bids\n");
        return -1;
    }

    int words=(goods+dummy-1)/(sizeof(int)<<3)+1;	// Determine how many ints are needed to represent goods;
    int regsize=(sizeof(int)+     // size of bid value
                 words*sizeof(int)+  // size of wanted goods
            //     sizeof(int)+        // size of dummy identifier
                 sizeof(int)         // size of bid identifier
                )/sizeof(int);


    int *data=(int*) malloc(bids*sizeof(int)*regsize);
    int bidid;
    unsigned int *bidset=(unsigned int*) malloc(sizeof(int)*words);

    problem->goods=goods+dummy;
    problem->bids=bids;
    problem->dummy=0;
    problem->data=data;
    problem->regsize=regsize;
    problem->words=words;

    while (!feof(fr)) {
        linenumber++;
        tmp=fgets(line,500,fr);
        pos=0;
        getSymbol(line,&pos,&a);
        if (a.utype==ERROR) return linenumber;
        if (a.utype != EMPTY) {
            //get bidid
            bidid=a.u.ival;
            //	*bidIdentifier(data,bidid,regsize)=bidid;
            *bidIdentifier(problem,bidid)=bidid;
            getSymbol(line,&pos,&a);

			if (intprices) *bidValue(problem,bidid)=(int) a.u.ival;  // get and store bid value		
			else *bidValue(problem,bidid)=(int) floor(a.u.fval);  

            bzero(bidset,words*sizeof(int));
         //   *bidDummyGood(problem,bidid)=0;
            while (a.utype != HASH) {
                getSymbol(line,&pos,&a);
                //if (a.u.ival>goods-1) *bidDummyGood(problem,bidid)=a.u.ival;
                //else 
				setBitOn(bidset,words,a.u.ival);
                count++;
            }
            memcpy(bidGoods(problem,bidid),bidset,words*sizeof(int));

        }
    }
    fclose(fr);
	free(bidset);
    return 0;
}
int dominates(problem_def * problem, int a, int b) {

    if ((*bidValue(problem,a)) < (*bidValue(problem,b))) return 0;
  /*  int dummy_a=*(bidDummyGood(problem,a));
    int dummy_b=*(bidDummyGood(problem,b));
    if (dummy_a) if (dummy_a != dummy_b) return 0; */
    unsigned int i,*ca,*cb,ret;
    ca=bidGoods(problem,a);
    cb=bidGoods(problem,b);
    ret=1;
    for (i=0; i<problem->words; i++) {
        if ((ca[i] & cb[i])!=ca[i]) ret=0;
    }
    return ret;
}
int removeDominates(problem_def * problem) {

    int i,j,k,c1=0,c2=0;

    unsigned int *gs=(unsigned int*) malloc(sizeof(int)*problem->words); // gs -> for store unitary bids
    for (i=0; i<problem->words; i++) gs[i]=0;	       // initialize gs
    for (i=0; i<problem->bids; i++) {
        if ((bidSize(problem,bidGoods(problem,i)))==1) {	// if unitary bid, save to gs
			for (k=0; k<(problem->words); k++) 
				gs[k]=gs[k]+(bidGoods(problem,i))[k];	        
				c1++;
		}
        for(j=i+1; j<problem->bids; j++)               // find dominates    
            if (dominates(problem,i,j)) {
                (*bidIdentifier(problem,j)) =-1;
                c2++;
            } else if (dominates(problem,j,i)) {
                (*bidIdentifier(problem,i)) =-1;
                c2++;
            }
    }

	for (i=0;i<problem->goods;i++) {
		if (checkBitOn(gs,problem->words,i)) printf("1");
		else printf("0");
	}
	printf("\n");
//	free(gs);
	
	int bidnumber=0;
	problem_def *newproblem = (problem_def * ) malloc(sizeof(problem_def));
	newproblem->data = (int *) malloc (sizeof(int)*(problem->regsize)*((problem->bids)+problem->dummy+problem->goods-c1));
    newproblem->regsize = problem->regsize;
	int lastbidid;
	for (i=0;i<problem->bids;i++) {
		if (*bidIdentifier(problem,i) >=0) {
			memcpy(bidValue(newproblem,bidnumber),bidValue(problem,i),sizeof(int)*(problem->regsize));
			bidnumber++;
			lastbidid=*bidIdentifier(problem,i);
		} 
	}
	for (i=0;i<problem->goods;i++) {
		if (!checkBitOn(gs,problem->words,i)) {
			*bidValue(newproblem,bidnumber) =0;
			lastbidid++;
			*bidIdentifier(newproblem,bidnumber)=-1;
			bzero(bidGoods(newproblem,bidnumber),newproblem->words*sizeof(int));
			setBitOn(bidGoods(newproblem,bidnumber),problem->words,i);
			//*bidDummyGood(newproblem,bidnumber)=0;
			
			bidnumber++;
		}
	} 

	for (i=0;i<problem->dummy;i++) {
		*bidValue(newproblem,bidnumber) =0;
		lastbidid++;
		*bidIdentifier(newproblem,bidnumber)=-1;
		bzero(bidGoods(newproblem,bidnumber),newproblem->words*sizeof(int));
	//	*bidDummyGood(newproblem,bidnumber)=problem->goods+i;
		bidnumber++;
	}
	problem->bids = bidnumber;
	free(problem->data);
	problem->data=newproblem->data;
	free(gs);	
	return c2;

}
void orderGoods(problem_def * problem) {
	int i,j,l,t,index,docont;
	double best=0.0;
	
	int *bidcount=(int *) malloc(sizeof(int)*(problem->goods));
	int *bidlenght=(int *) malloc(sizeof(int)*(problem->goods));
	unsigned int *listMap=(unsigned int *) malloc(sizeof(int)*(problem->goods));
	
	for (i=0;i<problem->goods;i++) {
		bidcount[i]=0;
		bidlenght[i]=0;
		listMap[i]= (unsigned int) -1;
	}
	double score;
	
	for (l=0;l<problem->goods;l++) {
		for (i=0;i<problem->goods;i++) {
			bidcount[i]=0;
			bidlenght[i]=0;
		}

		for (i=0;i<problem->bids;i++) {

			docont=0;
			for (j=0;j<problem->goods;j++) {
				if (checkBitOn(bidGoods(problem,i),problem->words,j)) {
					if (listMap[j]<l) docont=1;
				}
			}
			if (docont) continue;
			for (j=0;j<problem->goods;j++) {
				if (checkBitOn(bidGoods(problem,i),problem->words,j)) {
					bidcount[j]=bidcount[j]+1;
					bidlenght[j]=bidlenght[j]+bidSize(problem,bidGoods(problem,i));
				//	if (*bidDummyGood(problem,i)) bidlenght[j]++; 
				}

			}

		}
		best=100000.0;
		for (t=0;t<problem->goods;t++) {
			if (!bidcount[t]) continue;
			score = 1.0/((double) bidcount[t]*(double) bidlenght[t]/ (double) bidcount[t]);
			if (score < best) {
				best=score;
				index=t;
			}
		}
		listMap[index]=l;
	}
	
	unsigned int * newres= (unsigned int*) malloc(sizeof(int)*problem->words);
	for (i=0;i<problem->bids;i++) {
		for (j=0;j<problem->words;j++) newres[j]=0;
		for (j=0;j<problem->goods;j++) {
			if (checkBitOn(bidGoods(problem,i),problem->words,j))
				setBitOn(newres,problem->words,listMap[j]);
		}
		memcpy(bidGoods(problem,i),newres,sizeof(int)*problem->words);
	}

}

void orderGoodsNew(problem_def * problem) {
	int i,j,l,t,index,docont;
	double best=0.0;
	
	int *bidcount=(int *) malloc(sizeof(int)*(problem->goods));
	int *bidlenght=(int *) malloc(sizeof(int)*(problem->goods));
	unsigned int *listMap=(unsigned int *) malloc(sizeof(int)*(problem->goods));
	
	for (i=0;i<problem->goods;i++) {
		bidcount[i]=0;
		bidlenght[i]=0;
		listMap[i]= (unsigned int) -1;
	}
	double score;
	
	for (l=0;l<problem->goods;l++) {
		for (i=0;i<problem->goods;i++) {
			bidcount[i]=0;
			bidlenght[i]=0;
		}

		for (i=0;i<problem->bids;i++) {

			docont=0;
			for (j=0;j<problem->goods;j++) {
				if (checkBitOn(bidGoods(problem,i),problem->words,j)) {
					if (listMap[j]<l) docont=1;
				}
			}
			if (docont) continue;
			for (j=0;j<problem->goods;j++) {
				if (checkBitOn(bidGoods(problem,i),problem->words,j)) {
					bidcount[j]=bidcount[j]+1;
					bidlenght[j]=bidlenght[j]+bidSize(problem,bidGoods(problem,i));
				//	if (*bidDummyGood(problem,i)) bidlenght[j]++; 
				}

			}

		}
		best=100000.0;
		for (t=0;t<problem->goods;t++) {
			if (!bidcount[t]) continue;
			score = -1.0/((double) bidcount[t]);
			if (score < best) {
				best=score;
				index=t;
			}
		}
		listMap[index]=l;
	}
	
	unsigned int * newres= (unsigned int*) malloc(sizeof(int)*problem->words);
	for (i=0;i<problem->bids;i++) {
		for (j=0;j<problem->words;j++) newres[j]=0;
		for (j=0;j<problem->goods;j++) {
			if (checkBitOn(bidGoods(problem,i),problem->words,j))
				setBitOn(newres,problem->words,listMap[j]);
		}
		memcpy(bidGoods(problem,i),newres,sizeof(int)*problem->words);
	}

}

void CreateBins(problem_def * problem) {

    int i,j;


    problem_def *newproblem = (problem_def *) malloc(sizeof(problem_def));
    newproblem->goods=problem->goods;
    newproblem->bids=0;
    newproblem->dummy=problem->dummy;

    newproblem->regsize=problem->regsize;
    newproblem->words=problem->words;
    newproblem->data= (int *) malloc((problem->regsize)*sizeof(int)*(problem->bids+problem->goods));  
    //int lastbid=problem->bids;
	problem->bins=(int*) malloc (sizeof(int)*newproblem->goods);
	problem->binelements=(int*) malloc (sizeof(int)*newproblem->goods);
	int bincreated;
	
    for (i=0;i<problem->goods;i++) {
		bincreated=0;
		problem->bins[i]=0;
		problem->binelements[i]=0;		
		
		for (j=0;j<problem->bids;j++) {
			
			if (*bidIdentifier(problem,j) >=0)
			if (checkBitOn(bidGoods(problem,j),problem->words,i)) {
				memcpy(bidValue(newproblem,newproblem->bids),bidValue(problem,j),sizeof(int)*(problem->regsize));
				*bidIdentifier(problem,j)=-1;
				if (!bincreated) {
					bincreated=1;
					problem->bins[i]=newproblem->bids;					
				}
				problem->binelements[i]=problem->binelements[i]+1;
				newproblem->bids++;
			} 
		}
    }    
	free(problem->data);
	problem->data=newproblem->data;
	problem->bids=newproblem->bids;
	free(newproblem);



	for (i=0;i<problem->bids;i++) {
		printf("[%i] ",i);
		printBid(problem,i);
	}
    for (i=0;i<problem->goods;i++) {
		printf("[%2i]=>starts at %3i ends at %3i | ",i,problem->bins[i],problem->binelements[i]);
		if ((i%3) == 2) printf("\n");
		//printBid(problem,problem->bins[i]);
	} 
	printf("\n");  

}

void printBins(problem_def* problem) {
	int i,binname=0,end=0;
	int *tops = (int*) malloc(sizeof(int) * problem->goods);
	for (i=0;i<problem->goods;i++) tops[i]=0;
	
	// Header
	for (i=0;i<problem->goods;i++) {
		if (problem->binelements[i]>0) printf("%3i,",binname++);
	}
	printf("\n");
	while (!end) {
		end=1;
		for (i=0;i<problem->goods;i++) {
			if (problem->binelements[i]>0) {
				if (tops[i]<problem->binelements[i]) {
					printf("%3i,",*bidIdentifier(problem,problem->bins[i]+tops[i]));
					tops[i]++;
					end=0;
				} else 
				printf("   ,");
			}
		}
		printf("\n");
	}; 
	
	
}

binstr* initbinlist(problem_def *p,int * gpudata, int ingpu) {

	binstr *start = (binstr *) malloc(sizeof (binstr));
	binstr *current=start;
	start->prev=NULL;
	start->next=NULL;	
	start->binid=0;
	start->dfactor=0;
	start->elements=p->binelements[0];
	if (ingpu) start->ptr=&(gpudata[(p->bins[0])*(p->regsize)]);
	else start->ptr=&(p->data[(p->bins[0])*(p->regsize)]);	
	binstr *newbin;
	int i;
	for (i=1;i<=p->goods;i++) {
		if (p->binelements[i] > 0) {			
			newbin=(binstr *) malloc(sizeof (binstr));
			newbin->prev=current;
			newbin->next=NULL;
			newbin->binid=i;
			newbin->dfactor=0;
			newbin->elements=p->binelements[i];
			newbin->artificial=0;
			newbin->ratio=1.0;
			if (ingpu) newbin->ptr=&(gpudata[(p->bins[i])*(p->regsize)]);
			else newbin->ptr=&(p->data[(p->bins[i])*(p->regsize)]);
			current->next=newbin;
			current=newbin;
		}
			
	}
	return start;
}

binstr* freebinlist(binstr *b){
	//remove elment from list and return a ptr to the first element 

	binstr *ret;
	if (b->next!=NULL) b->next->prev=b->prev;
	if (b->prev!=NULL) b->prev->next=b->next;
	
	if ((b->prev==NULL) && (b->next==NULL)) ret = NULL;
	else {
		if (b->prev==NULL) ret=b->next;
		else {
			ret=b;		
			while(ret->prev!=NULL) ret=ret->prev;
		}				
	}
	free(b);
	return ret;

}

binstr* addbinlist(binstr *b,int size, int * result, int newid, double ratio) {

	binstr *newbin;
	if (b==NULL) {
		newbin = (binstr *) malloc (sizeof(binstr));
		newbin->prev = NULL;
		newbin->next = NULL;
		newbin->elements=size;
		newbin->binid=newid;
		newbin->artificial=1;
		newbin->ptr=result;
		newbin->ratio=ratio;
		
	} else {
		while (b->next !=NULL) b=b->next;
		newbin = (binstr *) malloc (sizeof(binstr));
		newbin->prev=b;
		b->next=newbin;
		newbin->next=NULL;
		newbin->ptr=result;
		newbin->elements=size;
		newbin->binid=newid;
		newbin->artificial=1;
		newbin->ratio=ratio;	
	}
	return newbin;
}

int binlistsize(binstr *b) {
	binstr *c=b;
	int ret=0;
        while (c->prev!=NULL) {c=c->prev;}
	while (c!=NULL) {c=c->next; ret+=1; }
	return ret;
}

void printbinids(binstr *b) {
	printf("Remaining bins: ");
	binstr *c=b;
        while (c->prev!=NULL) { c=c->prev;}
	while (c!=NULL) {printf("%i ",c->binid); c=c->next;}
	printf("\n");
}

int getBin(problem_def *p, int x) {
	int i;
	for(i=0;i< p->goods;i++) {
		if (checkBitOn(bidGoods(p,x),p->words,i)) return i;
	}
	return 0;
}

__global__ void dostuff(int* data, int regsize, int start1,int end1) {
	int pos=blockIdx.x*16+threadIdx.x;
	if (pos<end1)
		data[(pos+1)*(regsize)-1]=data[(pos+1)*(regsize)-1]+1;
}




__global__ void merge(int* data, int* data2,int *result, int regsize, int words, int start1,int end1, int start2, int end2) {
	int posx=blockIdx.x*blockDim.x+threadIdx.x;
	int posy=blockIdx.y*blockDim.y+threadIdx.y;
	int out=((end1)*(posy))+posx;
	int i,conflict=0;
	int remove1=1;
	int remove2=1;
	

	if ((posx >end1) || (posy >end2)) ; else 
	if ((posx==end1) && (posy==end2)) ; else {
		if (posx==end1) {remove1=0; out=((end1)*(end2))+posy; }
		if (posy==end2) {remove2=0; out=((end1)*(end2))+end2+posx; }
		for (i=0;i<words;i++) {
			if ((remove1*data[(start1+posx)*(regsize)+i+1]) & (remove2*data2[(start2+posy)*(regsize)+i+1])) conflict=1;
			result[(out)*(regsize)+i+1]= (remove1*data[(start1+posx)*(regsize)+i+1]) | (remove2*data2[(start2+posy)*(regsize)+i+1]);	
		}
		result[out*regsize+regsize-1]=out;
		if ((conflict)) result[(out)*(regsize)]=0;
		else result[(out)*(regsize)]= ((int) remove1*data [(start1+posx)*(regsize)]) + ((int) remove2*data2 [(start2+posy)*(regsize)]);
	}
}


__global__ void compactvsh(int *result, int regsize,int *compacted) {
	int blockDimXY = blockDim.x*blockDim.y;
	int threadInBlock = threadIdx.y * blockDim.x + threadIdx.x;
	int blockInGrid=(blockIdx.y * gridDim.x + blockIdx.x);	
	int threadInGrid = blockInGrid * blockDimXY + threadInBlock;
	int j;
	__shared__ int sumscan[1024];

	sumscan[threadInBlock] = (result[threadInGrid*regsize] == 0)?0:1;
	compacted[threadInGrid*regsize]=0;
	__syncthreads();
	int offset;
	for (offset = 1; offset < blockDimXY; offset *= 2) {
	  	  if ((threadInBlock - offset >= 0)) {
			j= sumscan[threadInBlock] + sumscan[threadInBlock - offset];
			__syncthreads();
			sumscan[threadInBlock]=j;
		 } else __syncthreads();
		__syncthreads();
	} 
	
	__syncthreads();

	if (threadInBlock==blockDimXY-1) {
		result[gridDim.x*gridDim.y*blockDimXY*regsize+blockInGrid]=sumscan[threadInBlock];
	}
	__syncthreads();

	int destreg=-1;

	if (threadInBlock==0)
			destreg=blockInGrid*blockDimXY+sumscan[threadInBlock]-1;
	else if (sumscan[threadInBlock-1]!=sumscan[threadInBlock]) 
			destreg=blockInGrid*blockDimXY+sumscan[threadInBlock]-1;

	if (destreg >= 0) {
		compacted[destreg*regsize]=result[threadInGrid*regsize];
		for (j=1;j<regsize;j++)
			compacted[destreg*regsize+j] = result[threadInGrid*regsize+j];
	}
}	


__global__ void joinbl(int* sizes,int regsize,int *compacted, int *joined){
		
	__shared__ int startpos;
	int threadInGrid=blockDim.x*blockIdx.x+threadIdx.x;
	int i;
	if (threadIdx.x==0) {
		startpos=0;	
		for (i=0;i<blockIdx.x;i++) {
			startpos+=sizes[i];
		}
	}
	__syncthreads();

        if (threadIdx.x<sizes[blockIdx.x]) {
		for (i=0;i<regsize;i++)	
			joined[(startpos+threadIdx.x)*regsize+i]=compacted[threadInGrid*regsize+i];
	}
}


__global__ void gpuremovedominates(int* data, int regsize, int words,int elements) {
   int posx=blockIdx.x*blockDim.x+threadIdx.x;
   int posy=blockIdx.y*blockDim.y+threadIdx.y;
   int i,dominated=1;
   if ((posx!=posy) && (posx < elements) && (posy < elements)) {
	if ((data[posx*regsize]>data[posy*regsize]) && (data[posy*regsize]>0)) {
		for(i=1;i<=words;i++) {
			if ((data[posx*regsize+i] & data[posy*regsize+i])!=data[posx*regsize+i] ) dominated=0;		
		}
		if (dominated) 	data[posy*regsize]=0;			
	}
   }
}


__global__ void gpusetzeros(int * data, int regsize, int elements, int vecsize) {
	

	int threadInGrid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x*blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int i;
	if ((threadInGrid >= elements) && (threadInGrid < vecsize)) {
		for (i=0;i<regsize;i++) data[threadInGrid*regsize+i]=0;
	}
	
}

int roundup1024(int v) {
   return (ceil(1.0*v/1024))*1024;
}



void p(int * gpumem,int sizev,const char * prefix,int regsize, int words) {
	
	int *bl = (int *) malloc(sizev*sizeof(int));
	gpuErrchk(cudaMemcpy(bl , gpumem, sizev*sizeof(int), cudaMemcpyDeviceToHost ));
	int i;
	for (i=0;i<sizev/regsize;i++) { 
		printf ("(%i)%s ",i,prefix); 
		printBidItemList(bl, regsize, words,i);		
	}
	free (bl);
} 

void p2(int * mem,int sizev,const char * prefix,int regsize, int words) {
	
	
	int i;
	for (i=0;i<sizev/regsize;i++) { 
		printf ("(%i)%s ",i,prefix); 
		printBidItemList(mem, regsize, words,i);		
	}
	
} 


void compresss_columns(problem_def *problem, int *data1,int* data2,int begin1, int count1, int begin2, int count2,int** retresult, int *retsize) {
	int firstNumberOfElements=count1;
	int secondNumberOfElements=count2;
	int regsize=problem->regsize;
	int words=problem->words;
	int sizeresult=(firstNumberOfElements+1)*(secondNumberOfElements+1)-1;
	float relation=1.0*secondNumberOfElements/firstNumberOfElements;
	int i;
	int blockx;
	int * blockcombinations,totalelements;

	int debug_compressing=0;



	if (relation < 0.001) blockx = 1024;
	else   if (relation < 0.004) blockx = 512;
	else if (relation < 0.0015) blockx = 256;
	else if (relation < 0.06) blockx = 128;
	else if (relation < 0.25) blockx = 64;
	else blockx = 32;
	int blocky=1024/blockx;
	
	
	dim3 dimBlock2(blockx,blocky);
	int gridx=(((count1 ) / blockx)+1 );
	int gridy=(((count2 ) / blocky)+1 );
	dim3 dimGrid2(gridx,gridy);
	dim3 dimBlock3(1024);
	dim3 dimGrid3(gridx*gridy);

	int *gpuresult,*gpucompacted,*gpubl;
	int sizeresult1024=gridx*gridy*1024; //roundup1024(sizeresult);

        if (debug_compressing & 1) {
		printf("************************ INFO compress columns ***************** \n");	
		printf("regsize %i, words %i, elements bins (%i,%i) begin=(%i,%i)\n",regsize,words,count1,count2,begin1,begin2);
		printf("sizeresult %i, sizeresult1024 %i, blockxy=(%i,%i) gridxy=(%i,%i)\n", sizeresult, sizeresult1024,blockx,blocky,gridx,gridy); 
		printf("**************************************************************** \n");	
	}

	gpuErrchk(cudaMalloc((void **) &gpucompacted,sizeresult1024*regsize*sizeof(int)));
	gpuErrchk(cudaMalloc((void **) &gpubl,(sizeresult1024*regsize+gridx*gridy)*sizeof(int)));
	gpuErrchk(cudaMalloc((void **) &gpuresult,(sizeresult1024*regsize+gridx*gridy)*sizeof(int)));
	
	merge<<<dimGrid2, dimBlock2>>>(data1,data2,gpuresult,regsize, words, begin1,count1,begin2,count2); 
	gpusetzeros<<<dimGrid2,dimBlock2>>>(gpuresult,regsize,sizeresult,sizeresult1024);
	if (debug_compressing & 2) p(gpuresult,sizeresult*regsize,"MERGE",regsize,words);		

	blockcombinations = (int*) malloc(gridx*gridy*sizeof(int));
	gpuErrchk(cudaMemcpy(blockcombinations , &gpuresult[sizeresult1024*(regsize)],gridx*gridy*sizeof(int), cudaMemcpyDeviceToHost ));

	compactvsh<<<dimGrid2, dimBlock2>>>(gpuresult,regsize,gpucompacted);

	blockcombinations = (int*) malloc(gridx*gridy*sizeof(int));
	gpuErrchk(cudaMemcpy(blockcombinations , &gpuresult[sizeresult1024*(regsize)],gridx*gridy*sizeof(int), cudaMemcpyDeviceToHost ));


	totalelements=0;
	for (i=0;i<gridx*gridy;i++) {
		totalelements+=blockcombinations[i];
		if (debug_compressing &2) printf("PH1 Block %i -> %i elements \n",i,blockcombinations[i]);	
	}
	int totalelements1024=roundup1024(totalelements);

	joinbl<<<dimGrid3,dimBlock3>>>(&gpuresult[sizeresult1024*regsize],regsize,gpucompacted,gpubl);

	dim3 dimBlock4(32,32);
	dim3 dimGrid4(totalelements1024/32,totalelements1024/32);	

	if (debug_compressing & 2) p(gpubl,totalelements*regsize,"BEFOREREMOVE",regsize,words);	
	gpuremovedominates<<<dimGrid4,dimBlock4>>>(gpubl,regsize,words,totalelements);

	if (debug_compressing & 2) p(gpubl,totalelements*regsize,"AFTERREMOVE",regsize,words);
	gpusetzeros<<<dimGrid4,dimBlock4>>>(gpubl,regsize,totalelements,sizeresult1024);
	
	dim3 dimBlock5(1024);
	dim3 dimGrid5(sizeresult1024/1024);
	
	if (debug_compressing & 2) p(gpubl,sizeresult1024*regsize,"SETTINGZEROS",regsize,words);

	compactvsh<<<dimGrid5,dimBlock5>>>(gpubl,regsize,gpucompacted);
	

	if (debug_compressing & 2) p(gpucompacted,totalelements*regsize,"COMPACTED",regsize,words);

	int *ans = (int * ) malloc(sizeof(int)*sizeresult1024/1024);

	cudaMemcpy(ans,&gpubl[sizeresult1024*regsize],sizeof(int)*(sizeresult1024/1024), cudaMemcpyDeviceToHost);

	*retsize=0;
	for (i=0;i<sizeresult1024/1024;i++) {
		*retsize+=ans[i];
		if (debug_compressing & 2) printf("PH2 Block %i -> %i elements \n",i,ans[i]);	
	}	

	joinbl<<<dimGrid3,dimBlock3>>>(&gpubl[sizeresult1024*regsize],regsize,gpucompacted,gpubl);

	*retresult=gpubl;
	free(blockcombinations);	
	free(ans);
	gpuErrchk(cudaFree(gpucompacted));
	gpuErrchk(cudaFree(gpuresult));	
		
}




void compress_seq(problem_def *problem, int *data1,int* data2,int begin1, int count1, int begin2, int count2,int** retresult, int *retsize) {
	
	int debug_seq=0;
	int sizeresult=((count1+1)*(count2+1)-1);
	
	int *result=(int *) malloc(sizeresult*(problem->regsize)*sizeof(int));
	int *compacted= (int *) malloc(sizeresult*(problem->regsize)*sizeof(int));
	int i,j,k,conflict;
	int *reg1,*reg2,*regdest;
	
	if (debug_seq) printf("sizeresult %i begins %i,%i counts %i,%i regsize %i words %i \n",sizeresult, begin1,begin2,count1,count2,problem->regsize, problem->words);

	for (i=0;i<count1;i++) {
		for (j=0;j<count2;j++) {
			regdest=&result[(j*count1+i)*(problem->regsize)];
			reg1=&data1[(begin1+i)*(problem->regsize)];
			reg2=&data2[(begin2+j)*(problem->regsize)];
			conflict=0;
			for (k=0;k<problem->words;k++) {
				if ((reg1[k+1]) & (reg2[k+1])) conflict=1;
				regdest[k+1] = reg1[k+1] | reg2[k+1];					
			}
			if (conflict) regdest[0] = 0;
			else regdest[0]= reg1[0] + reg2[0];
			
		}
	}
	for (j=0;j<count2;j++) {				
		regdest=&result[(problem->regsize)*(count1*(count2)+j)];
		reg2=&data2[(begin2+j)*(problem->regsize)];
		for (k=0;k<problem->regsize;k++) {
			regdest[k] = reg2[k];					
		}		
	}	
	for (j=0;j<count1;j++) {				
		regdest=&result[(problem->regsize)*((count1+1)*(count2)+j)];
		reg1=&data1[(begin1+j)*(problem->regsize)];
		for (k=0;k<problem->regsize;k++) {
			regdest[k] = reg1[k];					
		}		
	}
	
	if (debug_seq) {	

	printf("SIZE MERG-SEQ: %i\n",sizeresult);
	p2(result,sizeresult*(problem->regsize),"MERG-SEQ",problem->regsize,problem->words);
	}

	j=0;
        for (i=0;i<sizeresult;i++) {
		regdest=&compacted[j*(problem->regsize)];
		reg1=&result[i*(problem->regsize)];
                if (*reg1) {
			j++;
                        for (k=0;k<problem->regsize;k++) regdest[k]=reg1[k];			
		}
	} 

	if (debug_seq) {
	printf("SIZE COMP-SEQ: %i\n",j);
	p2(compacted,j*(problem->regsize),"COMP-SEQ",problem->regsize,problem->words);
	}


	int compsize=j;
	int dominated;
	for (i=0;i<compsize;i++) 
		for (j=0;j<compsize;j++) {
			dominated=1;
			reg1=&compacted[i*(problem->regsize)];
			reg2=&compacted[j*(problem->regsize)];
			if ((reg1!=reg2) && (*reg1>*reg2) && (*reg2>0)) {
				for (k=0;k<problem->words;k++)
					if ((reg1[k+1] & reg2[k+1])!= reg1[k+1])dominated=0;
				if (dominated) *reg2 = 0;
			}

		}

	if (debug_seq) {
	printf("SIZE REMD-SEQ: %i\n",j);
	p2(compacted,j*(problem->regsize),"REMD-SEQ",problem->regsize,problem->words);

	}

	sizeresult=j;
	j=0;
        for (i=0;i<sizeresult;i++) {
		regdest=&result[j*(problem->regsize)];
		reg1=&compacted[i*(problem->regsize)];
                if (*reg1) {
			j++;
                        for (k=0;k<problem->regsize;k++) regdest[k]=reg1[k];			
		}
	} 

	if (debug_seq) {
	printf("SIZE FIN-SEQ: %i\n",j);
	p2(result,j*(problem->regsize),"FIN-SEQ",problem->regsize,problem->words);
	}

	free(compacted);
        *retresult=result;
	*retsize=j;	
}


void writeProblem(binstr * b) {
	
}



int main(int argc, char *argv[]) {


    int seq=0;	
    double t0,t1;
    //int pos=0;
    int readerror=0;
    int rmv=0;
	int intprices;

	char filenameR[100];
	problem_def problem;
    srand(time(NULL));
    if (checkParameters(argc,argv,filenameR,&intprices)) {
        t0=sampleTime();
        if (readerror=readFile(&problem,filenameR,intprices)) {
            printf("Parse error, line %i \n",readerror);
        };
        printf("==== Auction started ====\n");
        printf("Goods: %i, Dummy: %i, Bids: %i\n",problem.goods,problem.dummy,problem.bids);
        t1=sampleTime();
        printf("File read %f sec.\n",t1-t0);
        
	t0=sampleTime();
//        rmv=removeDominates(&problem);
        t1=sampleTime();
        printf("Remove %i dominates: %f sec.\n",rmv, t1-t0);
		
	orderGoods(&problem);

	t0=sampleTime();
	CreateBins(&problem);
        t1=sampleTime();
        printf("Bin creation: %f sec.\n", t1-t0);	
	
	printBins(&problem);




	int *gpudata;
	int dsize=sizeof(int)*problem.regsize*problem.bids;
	t0=sampleTime();
	gpuErrchk(cudaMalloc( (void **) &gpudata,dsize));
	gpuErrchk(cudaMemcpy( gpudata, problem.data, dsize, cudaMemcpyHostToDevice )); 
        t1=sampleTime();
        printf("Time copy data to GPU %f sec.\n",t1-t0);


	//Merge two bins
		

	int * retresult,retsize;
	int higher_bin=0;
	binstr *src1;
	binstr *src2;
	int i=1;
	binstr * binlist;
	double ratio;

	if (seq) binlist = initbinlist(&problem,gpudata,0);
	else     binlist = initbinlist(&problem,gpudata,1);	
	binstr *binstart= binlist;
	printf("INITIAL SITUATION\n");
	higher_bin=binlist->binid;
	while (binlist->next!=NULL) {
		printf("Bin [%i] has %i bids\n",binlist->binid,binlist->elements);		
		binlist=binlist->next;
		higher_bin=(binlist->binid>higher_bin)?binlist->binid:higher_bin;
	}
	printf("Bin [%i] has %i bids\n",binlist->binid,binlist->elements);


	binstr * hardlist=NULL;	
	while ((binlistsize(binstart))>2) {

	printf(" ROUND %i %i bins\n",++i,binlistsize(binstart));	
	binlist=binstart;	
	while (binlist->next!=NULL) binlist=binlist->next;
	src1=NULL;
	src2=NULL;


	while ((binlist!=NULL)) {
		if (binlistsize(binstart)==1) {
			binlist=NULL;
			break;
		}
		if (src1==NULL)	src1=binlist;	
		else if (src2==NULL) src2=binstart;	
						

		if ((src1!=NULL) && (src2!=NULL)) {
			higher_bin+=1;
			printf("Merging bins %i and %i ",src1->binid,src2->binid);
		
			if (seq) {
				compress_seq(&problem,src1->ptr,src2->ptr,0, 
				src1->elements,0,src2->elements,&retresult,&retsize);			
			} else {
				compresss_columns(&problem,src1->ptr,src2->ptr,0,
				src1->elements,0,src2->elements,&retresult,&retsize);
			}
			
			ratio=1.0*retsize/((src1->elements+1)*(src2->elements+1)-1);
			printf("--> created bin %i of %i elements ratio %i/%i=%f\n",higher_bin,retsize,retsize,((src1->elements+1)*(src2->elements+1)-1),ratio);
			
			if (retsize>10000) {
				printf("adding bin %i to hardlist\n",higher_bin);
				hardlist=addbinlist(hardlist,retsize,retresult,higher_bin,ratio);
				
		//		addbinlist(src1,retsize,retresult,higher_bin);	
			} else addbinlist(src1,retsize,retresult,higher_bin,ratio);	
				
			binstart=freebinlist(src1);
			binstart=freebinlist(src2);
			src1=NULL;
			src2=NULL;
			printbinids(binstart);	

		}

		binlist=binlist->prev;

	}  // while ((binlist!=NULL))	
	} // while  ((binlistsize(binstart))>2)


	printf("exit\n");
        if (hardlist!=NULL) {
		binlist=binstart;
		while (binlist!=NULL) {	
			addbinlist(hardlist,binlist->elements,binlist->ptr,binlist->binid,binlist->ratio);
	       		binlist=binlist->next;	
	
		} 


		printf("bins in hardlist : ");
		binstart=hardlist;
        	while (binstart->prev!=NULL) { binstart=binstart->prev;}
		hardlist=binstart;
		while (hardlist!=NULL) {
			printf("%i[%ib.] ",hardlist->binid,hardlist->elements); 
			hardlist=hardlist->next;
		}
		printf("\n");


/*		while (hardlist->prev!=NULL); {
				if (hardlist->prev==NULL) printf("hardlist=NULL\n");
			hardlist=hardlist->prev;
			printf("@\n");
		}
		printf("@3\n");
		binstart=hardlist;
		while(hardlist!=NULL) {
			printf("Bin %i has %i bids\n",hardlist->binid, hardlist->elements);
			hardlist=hardlist->next;
			printf(".\n");		
		} */
	}
	gpuErrchk(cudaFree(retresult));	
	gpuErrchk(cudaFree(gpudata));

    }
    free(problem.data);  //sctucture with the data
    free(problem.bins);
    free(problem.binelements);
//	free(problem.binsBests);
    return EXIT_SUCCESS;
}

