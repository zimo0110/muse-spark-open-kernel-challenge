"""Decode diagnostic globaltimer records. No Torch, CUDA or submission I/O.

Milestone endpoints and per-CTA durations are distinct: phases overlap across
CTAs, so differences between global endpoints are not additive GEMM costs.
Keep nanosecond timestamps as Python integers until AFTER subtracting them.
Empty-ring measurements sample one wait in16; they are not total stall time.
Logical issue bytes are not measured DRAM traffic or achieved HBM bandwidth.
"""
import math

def percentiles(values):
    v=sorted(values)
    if not v:return None
    return [round(v[min(len(v)-1,int((len(v)-1)*p))],3) for p in (.5,.95,1.)]

def decode(rows,wgs):
    rows=[[int(v) for v in r] for r in rows]
    assert rows and all(len(r)==40 for r in rows)
    start=min(r[0] for r in rows if r[0])
    markers={}
    for k,name in ((1,'route'),(2,'G1'),(3,'G2'),(4,'barrier'),(5,'final')):
        stamps=[r[k] for r in rows if r[k]]
        if stamps:markers[name]=[round((min(stamps)-start)/1000,3),round((max(stamps)-start)/1000,3)]
    intervals=[('route',0,1),('G1',6 if wgs else 1,2),('G2',2,3),('G2_to_barrier',3,4),('final',4,5)]
    if wgs:intervals.extend([('route_to_gather',1,7),('gather_to_join',7,6)])
    durations={name:percentiles([(r[b]-r[a])/1000 for r in rows if r[a] and r[b]]) for name,a,b in intervals}
    for name,a,b in intervals:
        assert all(r[b]>=r[a] for r in rows if r[a] and r[b]),('nonmonotonic phase',name)
    result={'milestones_min_max_us':markers,'CTA_duration_p50_p95_max_us':durations}
    if wgs:
        producers={}
        for off,name in ((8,'weight_A'),(16,'weight_B'),(24,'activation_A'),(32,'activation_B')):
            v=[r[off:off+8] for r in rows if r[off] and r[off+1]]
            if not v:continue
            assert all(q[1]>=q[0] for q in v)
            begin=min(q[0] for q in v);end=max(q[1] for q in v)
            producers[name]={
                'issue_window_us':round((end-begin)/1000,3),
                'logical_MiB':round(sum(q[6] for q in v)/(1024**2),3),
                'stage_count':sum(q[7] for q in v),
                'empty_wait_sample_mean_us_p50_p95_max':percentiles([q[2]/q[3]/1000 for q in v if q[3]]),
                'dependency_wait_us_p50_p95_max':percentiles([q[4]/1000 for q in v]),
                'dependency_wait_events':sum(q[5] for q in v),
            }
        result['producer'] = producers
    return result
