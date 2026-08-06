#!/usr/bin/env python3
import json, pathlib, statistics, subprocess, tempfile
from datetime import date
ROOT=pathlib.Path(__file__).resolve().parents[1]; PATCH=ROOT/'patches/0001-control-q4k-moe-dual-swiglu.patch'; OUT=ROOT/'benchmark/results/moe-down-overlap-trace-hotpath-20260807.json'
t=PATCH.read_text(); needle='static const bool trace_moe_down_skip = []'; assert t.count(needle)==1
b=t[t.index(needle):t.index(needle)+600]; assert 'if (trace_moe_down_skip)' in b and 'once.fetch_add(1, std::memory_order_relaxed)' in b and 'skip: buffer overlap' in b
src=r'''#include <atomic>
#include <chrono>
#include <cstdio>
static std::atomic<int> once{1};
template<bool traced> double run(){constexpr int n=100000000; auto s=std::chrono::steady_clock::now(); for(int i=0;i<n;++i){if constexpr(traced){if(once.fetch_add(1,std::memory_order_relaxed)==0) asm volatile("");} asm volatile("":::"memory");} return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-s).count()/n;}
int main(){for(int i=0;i<9;++i) std::printf("%.9f %.9f\n",run<true>(),run<false>());}'''
with tempfile.TemporaryDirectory() as td:
 p=pathlib.Path(td)/'b.cpp'; e=pathlib.Path(td)/'b'; p.write_text(src); subprocess.run(['c++','-O3','-std=c++17',str(p),'-o',str(e)],check=True); rows=[tuple(map(float,x.split())) for x in subprocess.check_output([str(e)],text=True).splitlines()]
o=[x[0] for x in rows]; n=[x[1] for x in rows]; om=statistics.median(o); nm=statistics.median(n)
r={'date':str(date.today()),'iterations_per_trial':100000000,'trials':len(rows),'old_atomic_check_ns_median':om,'new_trace_disabled_ns_median':nm,'removed_ns_per_overlap_rejection':om-nm,'speedup_of_diagnostic_check':om/nm,'patch_integrity':'passed','scope':'synthetic host overlap-rejection control-path benchmark; not end-to-end inference'}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(OUT)
