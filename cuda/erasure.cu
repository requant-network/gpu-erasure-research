#include <stdint.h>
__device__ uint8_t serial(uint8_t a,uint8_t b){unsigned r=0,x=a;for(int bit=0;bit<8;bit++){if(b&(1<<bit))r^=x;x<<=1;if(x&256)x^=0x11d;}return r;}
__device__ unsigned long long clmul(unsigned long long a,unsigned long long b,unsigned long long c=0){unsigned long long r;asm("clmad.lo.u64 %0, %1, %2, %3;":"=l"(r):"l"(a),"l"(b),"l"(c));return r;}
__device__ unsigned long long spread(uint32_t x){return (x&255ULL)|((x&0xff00ULL)<<8)|((x&0xff0000ULL)<<16)|((x&0xff000000ULL)<<24);}
__device__ uint32_t reduce(unsigned long long x){
 const unsigned long long low=0x00ff00ff00ff00ffULL;
 x=(x&low)^clmul((x>>8)&low,0x1d);
 x=(x&low)^clmul((x>>8)&low,0x1d);
 return (x&255ULL)|((x>>8)&0xff00ULL)|((x>>16)&0xff0000ULL)|((x>>24)&0xff000000ULL);
}
extern "C" __global__ void validate_product(uint8_t* out){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<65536)out[i]=reduce(clmul(i>>8,i&255));}
extern "C" __global__ void encode_serial(const uint8_t* data,const uint8_t* coefficients,uint8_t* out,int k,int rows,int size){
 int i=blockIdx.x*blockDim.x+threadIdx.x;int row=blockIdx.y;if(i>=size)return;uint8_t r=0;
 for(int j=0;j<k;j++)r^=serial(data[(size_t)j*size+i],coefficients[row*k+j]);out[(size_t)row*size+i]=r;
}
extern "C" __global__ void encode_table(const uint8_t* data,const uint8_t* coefficients,const uint8_t* table,uint8_t* out,int k,int rows,int size){
 int i=blockIdx.x*blockDim.x+threadIdx.x;int row=blockIdx.y;if(i>=size)return;uint8_t r=0;
 for(int j=0;j<k;j++)r^=table[(int)coefficients[row*k+j]*256+data[(size_t)j*size+i]];out[(size_t)row*size+i]=r;
}
extern "C" __global__ void encode_nibble4(const uint8_t* data,const uint8_t* nibble,uint8_t* out,int k,int rows,int size){
 int i=blockIdx.x*blockDim.x+threadIdx.x;int first=blockIdx.y*4;if(i>=size)return;uint8_t r[4]={0,0,0,0};
 for(int j=0;j<k;j++){uint8_t x=data[(size_t)j*size+i];for(int z=0;z<4;z++)if(first+z<rows){const uint8_t* t=nibble+((first+z)*k+j)*32;r[z]^=t[x&15]^t[16+(x>>4)];}}
 for(int z=0;z<4;z++)if(first+z<rows)out[(size_t)(first+z)*size+i]=r[z];
}
extern "C" __global__ void encode_clmad4(const uint8_t* data,const uint8_t* coefficients,uint8_t* out,int k,int rows,int size){
 int i=(blockIdx.x*blockDim.x+threadIdx.x)*4;int first=blockIdx.y*4;if(i>=size)return;unsigned long long r[4]={0,0,0,0};
 for(int j=0;j<k;j++){
   uint32_t x=0;if(i+4<=size&&!(size&3))x=*(const uint32_t*)(data+(size_t)j*size+i);else for(int b=0;b<4&&i+b<size;b++)x|=(uint32_t)data[(size_t)j*size+i+b]<<(8*b);
   unsigned long long value=spread(x);for(int z=0;z<4;z++)if(first+z<rows)r[z]=clmul(value,coefficients[(first+z)*k+j],r[z]);
 }
 for(int z=0;z<4;z++)if(first+z<rows){uint32_t value=reduce(r[z]);if(i+4<=size&&!(size&3))*(uint32_t*)(out+(size_t)(first+z)*size+i)=value;else for(int b=0;b<4&&i+b<size;b++)out[(size_t)(first+z)*size+i+b]=value>>(8*b);}
}
extern "C" __global__ void gather(const uint8_t* data,const uint8_t* parity,const int* indices,uint8_t* survivors,int k,int size){
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(size_t)k*size)return;int row=t/size,offset=t%size,source=indices[row];survivors[t]=source<k?data[(size_t)source*size+offset]:parity[(size_t)(source-k)*size+offset];
}
extern "C" __global__ void restore(const uint8_t* survivors,const uint8_t* recovered,const int* map,uint8_t* out,int k,int size){
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(size_t)k*size)return;int row=t/size,offset=t%size,source=map[row];out[t]=source>=0?survivors[(size_t)source*size+offset]:recovered[(size_t)(-source-1)*size+offset];
}
