// Original research harness; links to the unmodified pinned ISA-L library.
#include <erasure_code.h>
#include <cstdint>
#include <cstring>
#include <vector>
extern "C" void requant_isal_tables(int k,int rows,unsigned char* coefficients,unsigned char* tables){ec_init_tables(k,rows,coefficients,tables);}
extern "C" void requant_isal_encode(int k,int rows,int size,unsigned char* tables,unsigned char* data,unsigned char* out){
 std::vector<unsigned char*> source(k),dest(rows);for(int i=0;i<k;i++)source[i]=data+(size_t)i*size;for(int i=0;i<rows;i++)dest[i]=out+(size_t)i*size;
 ec_encode_data(size,k,rows,tables,source.data(),dest.data());
}
extern "C" void requant_isal_pipeline(int k,int rows,int size,unsigned char* encode_tables,unsigned char* recovery_tables,const int* survivors,const int* missing,unsigned char* data,unsigned char* parity,unsigned char* recovered,unsigned char* restored){
 std::vector<unsigned char*> input(k),output(rows),available(k),repair(rows);
 for(int i=0;i<k;i++)input[i]=data+(size_t)i*size;
 for(int i=0;i<rows;i++){output[i]=parity+(size_t)i*size;repair[i]=recovered+(size_t)i*size;}
 ec_encode_data(size,k,rows,encode_tables,input.data(),output.data());
 for(int i=0;i<k;i++)available[i]=survivors[i]<k?input[survivors[i]]:output[survivors[i]-k];
 ec_encode_data(size,k,rows,recovery_tables,available.data(),repair.data());
 for(int i=0;i<k;i++){
   int index=-1;for(int j=0;j<rows;j++)if(missing[j]==i)index=j;
   std::memcpy(restored+(size_t)i*size,index>=0?repair[index]:input[i],size);
 }
}
