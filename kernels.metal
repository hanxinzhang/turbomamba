// Mamba-3-derived parameterization: Copyright (c) 2026, Dao AI Lab, Goombalab.
// Apache-2.0; see THIRD_PARTY_NOTICES.md. Modified for FP32 Apple MPS.

#include <metal_stdlib>
using namespace metal;

// Exactly one complete row per threadgroup. Width is 64 or 128, so all SIMD
// groups are full. No early exit inside a participating threadgroup.
kernel void rms_inference(
    const device float* x [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint i [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    if (row >= rows) return;
    threadgroup float partials[4];
    float value = x[row * width + i];
    float subtotal = simd_sum(value * value);
    if (i % 32 == 0) partials[i / 32] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (uint j = 0; j < width / 32; ++j) total += partials[j];
    float inverse = rsqrt(total / float(width) + 1.e-5f);
    // Read completion: safe even if a future fusion reuses this scratch.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    y[row * width + i] = value * inverse * weight[i];
}

// projected is [..., 2 * ff], ordered [value branch, gate branch], matching
// common.SwiGLU. Bounds cover every FF width, including non-multiples of 64.
kernel void gate_inference(
    const device float* projected [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& ff [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= rows * ff) return;
    uint row = id / ff;
    uint i = id % ff;
    float value = projected[row * 2 * ff + i];
    float gate = projected[row * 2 * ff + ff + i];
    float activation = gate / (1.f + exp(-gate));
    y[id] = value * activation;
}

kernel void prep(const device float* x [[buffer(0)]],const device float* dtb [[buffer(1)]],const device float* qb [[buffer(2)]],const device float* kb [[buffer(3)]],const device float* qw [[buffer(4)]],const device float* kw [[buffer(5)]],const device float* vx [[buffer(6)]],const device float* zx [[buffer(7)]],const device float* oldangle [[buffer(8)]],device float* q [[buffer(9)]],device float* k [[buffer(10)]],device float* v [[buffer(11)]],device float* z [[buffer(12)]],device float* coeff [[buffer(13)]],device float* angle [[buffer(14)]],constant uint& rank [[buffer(15)]],constant uint& batch [[buffer(16)]],uint id [[thread_position_in_grid]]) {
 if(id>=batch*4*rank*64)return;
 uint i=id%64,r=(id/64)%rank,h=(id/(64*rank))%4,b=id/(64*rank*4);
 uint stride=512+128*rank+12+16,base=b*stride;
 uint bo=base+512+r*64,qo=base+512+64*rank+r*64;
 uint dto=base+512+128*rank;
 float a=x[dto+4+h],d=x[dto+h]+dtb[h];float dt=max(d,0.f)+log(1.f+exp(-abs(d)));
 float decay=-max(a,0.f)-1.f/(1.f-min(a,0.f));decay=min(decay,-1.e-4f);
 float alpha=exp(dt*decay),trap=1.f/(1.f+exp(-x[dto+8+h]));
 if(r==0&&i==0){coeff[(b*4+h)*3]=alpha;coeff[(b*4+h)*3+1]=(1.f-trap)*dt*alpha;coeff[(b*4+h)*3+2]=trap*dt;}
 float qs=0.f,ks=0.f;for(uint j=0;j<64;j++){qs+=x[qo+j]*x[qo+j];ks+=x[bo+j]*x[bo+j];}
 float qn=rsqrt(qs/64.f+1.e-5f),kn=rsqrt(ks/64.f+1.e-5f);
 uint pair=i,ai=0;bool rotated=false,positive=true;
 if(rank>1){if(i<16){pair=i+32;ai=i;rotated=true;}else if(i>=32&&i<48){pair=i-32;ai=i-32;rotated=true;positive=false;}}
 else if(i<32){pair=i^1;ai=i/2;rotated=true;positive=(i%2==0);}
 float phase=0.f;if(rotated)phase=oldangle[(b*4+h)*16+ai]+tanh(x[dto+12+ai])*dt*3.14159265358979323846f;
 if(r==0&&i<16)angle[(b*4+h)*16+i]=oldangle[(b*4+h)*16+i]+tanh(x[dto+12+i])*dt*3.14159265358979323846f;
 uint bias=(h*rank+r)*64;
 float qc=x[qo+i]*qn*qw[i]+qb[bias+i],kc=x[bo+i]*kn*kw[i]+kb[bias+i];
 if(rotated){float qp=x[qo+pair]*qn*qw[pair]+qb[bias+pair],kp=x[bo+pair]*kn*kw[pair]+kb[bias+pair];float sn=sin(phase),cs=cos(phase);qc=qc*cs+(positive?-qp:qp)*sn;kc=kc*cs+(positive?-kp:kp)*sn;}
 q[id]=qc;k[id]=kc;uint pj=(h*rank+r)*64+i;
 v[id]=x[base+256+h*64+i]*vx[pj];float zg=x[base+h*64+i]*zx[pj];z[id]=zg/(1.f+exp(-zg));
}
kernel void step(const device float* q [[buffer(0)]],const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],const device float* z [[buffer(3)]],const device float* coeff [[buffer(4)]],const device float* prevk [[buffer(5)]],const device float* prevv [[buffer(6)]],device float* state [[buffer(7)]],const device float* skip [[buffer(8)]],const device float* outw [[buffer(9)]],device float* out [[buffer(10)]],constant uint& rank [[buffer(11)]],constant uint& batch [[buffer(12)]],uint id [[thread_position_in_grid]]) {
 if(id>=batch*4*64)return;uint p=id%64,h=(id/64)%4,b=id/256,bh=b*4+h;
 float alpha=coeff[bh*3],beta=coeff[bh*3+1],gamma=coeff[bh*3+2];float y[4]={0.f,0.f,0.f,0.f};
 for(uint n=0;n<64;n++){float curr=0.f,prev=0.f;for(uint r=0;r<rank;r++){uint base=(bh*rank+r)*64;curr+=k[base+n]*v[base+p];prev+=prevk[base+n]*prevv[base+p];}uint si=(bh*64+p)*64+n;float s=alpha*state[si]+beta*prev+gamma*curr;state[si]=s;for(uint r=0;r<rank;r++)y[r]+=s*q[(bh*rank+r)*64+n];}
 float value=0.f;for(uint r=0;r<rank;r++){uint bi=(bh*rank+r)*64+p;value+=(y[r]+skip[h]*v[bi])*z[bi]*outw[(h*rank+r)*64+p];}out[id]=value;
}
