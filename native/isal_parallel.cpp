// Original full-pipeline chunking harness; ISA-L arithmetic is unchanged.
#include <erasure_code.h>
#include <cstring>
#include <vector>
extern "C" void requant_isal_parallel(int k,int rows,int size,int threads,unsigned char* encode_tables,unsigned char* recovery_tables,const int* survivors,const int* missing,unsigned char* data,unsigned char* parity,unsigned char* recovered,unsigned char* restored){
 constexpr int CHUNK=262144;
 #pragma omp parallel num_threads(threads)
 {
  std::vector<unsigned char*> input(k),output(rows),available(k),repair(rows);
  #pragma omp for schedule(static)
  for(int offset=0;offset<size;offset+=CHUNK){
   int length=offset+CHUNK<=size?CHUNK:size-offset;
   for(int i=0;i<k;i++)input[i]=data+(size_t)i*size+offset;
   for(int i=0;i<rows;i++){output[i]=parity+(size_t)i*size+offset;repair[i]=recovered+(size_t)i*size+offset;}
   ec_encode_data(length,k,rows,encode_tables,input.data(),output.data());
   for(int i=0;i<k;i++)available[i]=survivors[i]<k?input[survivors[i]]:output[survivors[i]-k];
   ec_encode_data(length,k,rows,recovery_tables,available.data(),repair.data());
   for(int i=0;i<k;i++){int index=-1;for(int j=0;j<rows;j++)if(missing[j]==i)index=j;std::memcpy(restored+(size_t)i*size+offset,index>=0?repair[index]:input[i],length);}
  }
 }
}
