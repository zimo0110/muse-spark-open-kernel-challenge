"""Summarize okc diagnostic log records without interpreting GB10 as H200.

Usage: python3 tools/analyze_phase_probe.py /path/to/okc-status.json
Milestones overlap; producer logical bytes are not physical HBM measurements.
"""
import json,statistics,sys
from pathlib import Path

def parse(log):
 records=[]
 for line in log.splitlines():
  if line.startswith('[phase-probe] {'):
   try:records.append(json.loads(line.removeprefix('[phase-probe] ')))
   except json.JSONDecodeError:continue
 return records

def summarize(records):
 groups={}
 for q in records:groups.setdefault(q['M'],[]).append(q)
 result=[]
 for m,rows in sorted(groups.items()):
  row={'M':m,'seeds':len(rows),
       'control_event_us':round(statistics.median(q['cold_graph_us']['control'] for q in rows),3),
       'probes_within_3pct':all(all(q['overhead_within_3pct'].values()) for q in rows),
       'probe_over_control':{name:[q['probe_over_control'][name] for q in rows if name in q['probe_over_control']] for name in sorted({k for q in rows for k in q['probe_over_control']})}}
  phases=rows[0]['consumer']['CTA_duration_p50_p95_max_us']
  row['consumer_CTA_us']={name:{'median_of_p50':round(statistics.median(q['consumer']['CTA_duration_p50_p95_max_us'][name][0] for q in rows),3),
                             'largest_p95':max(q['consumer']['CTA_duration_p50_p95_max_us'][name][1] for q in rows),
                             'largest_max':max(q['consumer']['CTA_duration_p50_p95_max_us'][name][2] for q in rows)} for name in phases}
  if all('producer' in q for q in rows):
   row['producer']={}
   for name in ('weight_A','weight_B','activation_A','activation_B'):
    v=[q['producer'][name] for q in rows]
    row['producer'][name]={
      'logical_MiB_per_seed':[q['logical_MiB'] for q in v],
      'issue_window_us_per_seed':[q['issue_window_us'] for q in v],
      'largest_dependency_wait_p95_us':max(q['dependency_wait_us_p50_p95_max'][1] for q in v),
      'sampled_empty_wait_mean_p95_us_per_seed':[q['empty_wait_sample_mean_us_p50_p95_max'][1] if q['empty_wait_sample_mean_us_p50_p95_max'] else None for q in v]}
  result.append(row)
 return result

def main():
 j=json.loads(Path(sys.argv[1]).read_text());r=j.get('result') or {}
 records=parse(r.get('log_tail',''))
 out={'job':j.get('id'),'status':j.get('status'),'passed':r.get('passed'),'speedup_unchanged_timed_path':r.get('speedup'),
      'diagnostic_records':len(records),'expected_records':8,'summary':summarize(records),
      'cautions':['Milestones overlap across CTAs; these are not additive phase costs.',
                  'Logical bytes issued are not measured HBM traffic.',
                  'Empty-ring waits sample one in16; they are not total stall time.',
                  'Probe durations are unperturbed only if paired overhead supports that interpretation.']}
 print(json.dumps(out,indent=2))

if __name__=='__main__':main()
