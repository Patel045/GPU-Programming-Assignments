#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


using namespace std;

//*******************************************

#define MAX_TANKS 1000

__global__
void dkernel(int *xcoord, int *ycoord, int *score, int T, int H){
    int id = threadIdx.x;
    __shared__ int counter;
    __shared__ bool been_hit[MAX_TANKS];
    __shared__ int k;
    // __shared__ int xcoord[MAX_TANKS];
    // __shared__ int ycoord[MAX_TANKS];
    // __shared__ int score[MAX_TANKS];
    __shared__ int hp[MAX_TANKS];
    // xcoord[id] = d_xcoord[id];
    // ycoord[id] = d_ycoord[id];
    hp[id] = H;
    score[id] = 0;
    if(id==0){
        counter = T;
        k = 1;
    }
    int hit_id;
    int hit_dist;
    int cur_dist;
    int myx, myy, yourx, youry, curx, cury;
    __syncthreads();
    while(counter>1){
        hit_id = -1;
        hit_dist = INT_MAX;
        been_hit[id] = false;
        myx = xcoord[id];
        myy = ycoord[id];
        yourx = xcoord[(id+k)%T];
        youry = ycoord[(id+k)%T];
        if(hp[id] > 0){
            for(int j=0; j<T; ++j){
                curx = xcoord[j];
                cury = ycoord[j];
                if(hp[j]>0 && j!=id && ((long long)(myy-cury))*((long long)(myx-yourx))==((long long)(myx-curx))*((long long)(myy-youry))){
                    if(((long long)(yourx-myx))*((long long)(curx-myx))>=0ll && ((long long)(youry-myy))*((long long)(cury-myy))>=0ll){
                        cur_dist = abs(myy-cury)+abs(myx-curx);
                        if(cur_dist<hit_dist){ 
                            hit_dist = cur_dist;
                            hit_id = j;
                        }
                    }
                }
            }
        }
        __syncthreads();
        if(hit_id != -1){
            // printf("%d hits %d at %d\n", id, hit_id, k);
            score[id] += 1;
            been_hit[hit_id] = true;
            atomicSub(&hp[hit_id], 1);
            // printf("hp[%d] : %d\n", id, hp[id]);
        }
        // printf("been_hit[%d] = %d , hp[%d] = %d\n", id, been_hit[id], id, d_hp[id]);
        __syncthreads();
        if(been_hit[id] && hp[id]<=0){
            atomicSub(&counter,1);
        }
        if(id==0){
            k += 1;
            if(k%T==0) k += 1;
        }
        __syncthreads();
    }
    // if(id==0){
    //     printf("Rounds k : %d\n", k);
    // }
    // d_score[id] = score[id];
}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *d_xcoord;
    cudaMalloc(&d_xcoord, T * sizeof(int));
    cudaMemcpy(d_xcoord, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    int *d_ycoord;
    cudaMalloc(&d_ycoord, T * sizeof(int));
    cudaMemcpy(d_ycoord, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);
    int *d_score;
    cudaMalloc(&d_score, T * sizeof(int));

    cudaDeviceSynchronize();
    // cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferShared);

    dkernel<<<1,T>>>(d_xcoord, d_ycoord, d_score, T, H);
    

    cudaMemcpy(score, d_score, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // // free(hp);
    // cudaFree(d_xcoord);
    // cudaFree(d_ycoord);
    // // cudaFree(d_hp);
    // cudaFree(d_score);
    // cudaDeviceSynchronize();

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}