"""CPU-only frozen gold features and paired exploratory pilot leaf tables."""
import collections
import copy
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attncopy.paths import model_path
from scripts.prepare import ROOT,digest,dump
from scripts.resume_pilot import SNAP,OUTPUT as RECOVERY,read,merge_rows,recovery_plan,ARMS
from scripts.finish_pilot import OUT as FINAL_RECOVERY,merged_pilot
from attncopy.argument_features import string_features,walk,TOKEN_BINS,FRAGMENT_BINS,TYPE_BINS
from attncopy.scoring_review import max_assignment,reviewed_score
from attncopy.metrics import canonical
from tokenizers import Tokenizer


def prediction_leaves(record,row,payload=False):
    raw=row['raw'].strip()
    if payload:
        for opening in ['```json\n','```\n']:
            if raw.startswith(opening) and raw.endswith('\n```'):
                raw=raw[len(opening):-4]
                break
    def pairs(items):
        if len({k for k,_ in items})!=len(items):raise ValueError('duplicate keys')
        return dict(items)
    try:
        pred=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        json.dumps(pred,allow_nan=False)
        if not isinstance(pred,list) or not pred or not all(isinstance(p,dict) and isinstance(p.get('name'),str) and isinstance(p.get('arguments'),dict) for p in pred):return {}
    except (ValueError,TypeError):return {}
    gold=record['answers'];matched={}
    for name in sorted({g['name'] for g in gold}):
        gi=sorted([i for i,g in enumerate(gold) if g['name']==name],key=lambda i:canonical([gold[i]]))
        pi=sorted([i for i,p in enumerate(pred) if p['name']==name],key=lambda i:canonical([pred[i]]))
        n=max(len(gi),len(pi));weights=[[0]*n for _ in range(n)]
        for i,g in enumerate(gi):
            gl=dict(walk(gold[g]['arguments']))
            for j,p in enumerate(pi):
                pl=dict(walk(pred[p]['arguments']))
                count=sum(k in pl and type(pl[k]) is type(v) and pl[k]==v for k,v in gl.items())
                weights[i][j]=count*(n+1)**2+(canonical([gold[g]])==canonical([pred[p]]))*(n+1)+(set(gold[g]['arguments'])==set(pred[p]['arguments']))
        for i,j in max_assignment(weights).items():
            if i<len(gi) and j<len(pi):matched[gi[i]]=dict(walk(pred[pi[j]]['arguments']))
    correct={}
    for i,g in enumerate(gold):
        pl=matched.get(i,{})
        for path,value in walk(g['arguments']):
            correct[i,path]=path in pl and type(pl[path]) is type(value) and pl[path]==value
    return correct


def table(items,group,bins,arms_correct):
    result={}
    for label in bins:
        selected=[item for item in items if item[group]==label]
        keys=[(item['id'],item['call'],tuple(item['path'])) for item in selected]
        counts={arm:sum(arms_correct[arm].get(k,False) for k in keys) for arm in ARMS}
        paired={}
        for baseline in ['greedy','match-forced']:
            a=[arms_correct['attention+match-forced'].get(k,False) for k in keys]
            b=[arms_correct[baseline].get(k,False) for k in keys]
            gains=sum(x and not y for x,y in zip(a,b));losses=sum(y and not x for x,y in zip(a,b))
            paired[baseline]={'wrong_to_correct':gains,'correct_to_wrong':losses,
                              'both_correct':sum(x and y for x,y in zip(a,b)),
                              'both_wrong':sum(not x and not y for x,y in zip(a,b)),
                              'net_change':gains-losses,'net_change_over_n':(gains-losses)/len(keys) if keys else None}
        result[label]={'string_leaf_n':len(keys),'example_n':len({item['id'] for item in selected}),
                       'small_n':len(keys)<20,'correct':counts,'argument_em':{a:counts[a]/len(keys) if keys else None for a in ARMS},
                       'paired_attention_vs':paired,'duplicate_name_gold_leaf_n':sum(item['duplicate_name'] for item in selected)}
    return result


def main():
    assert (FINAL_RECOVERY/'complete.json').exists()
    recovery_plan()
    cfg=json.loads((ROOT/'configs/pilot.json').read_text())
    tok=Tokenizer.from_file(str(Path(model_path(cfg))/'tokenizer.json'))
    output={'scope':'exploratory unofficial implementation pilot; not official full benchmark',
            'feature_plan_sha256':digest(ROOT/'docs/ARGUMENT_SUBGROUP_PLAN.md'),'script_sha256':digest(__file__),
            'tokenizer_sha256':digest(Path(model_path(cfg))/'tokenizer.json'),
            'raw_sha256':{'dev':digest(SNAP/'runs/pilot-20261001/dev-predictions.jsonl'),
                          'test_segment1':digest(SNAP/'runs/pilot-20261001/test-predictions.jsonl'),
                          'test_segment2':digest(RECOVERY/'segment-2.jsonl'),
                          'test_segment3':digest(FINAL_RECOVERY/'segment-3.jsonl')},'command':sys.argv,'splits':{}}
    for split in ['dev','test']:
        records=read(SNAP/'runs/pilot-20261001'/f'{split}.jsonl')
        rows=read(SNAP/'runs/pilot-20261001/dev-predictions.jsonl') if split=='dev' else merged_pilot()[1]
        recmap={r['id']:r for r in records}
        features=[];non_string=0
        for rec in records:
            names=collections.Counter(g['name'] for g in rec['answers'])
            for i,g in enumerate(rec['answers']):
                for path,value in walk(g['arguments']):
                    if not isinstance(value,str):non_string+=1;continue
                    features.append({'id':rec['id'],'call':i,'path':list(path),'duplicate_name':names[g['name']]>1,
                                     **string_features(value,len(tok.encode(value,add_special_tokens=False).ids))})
        split_out={'string_leaf_n':len(features),'non_string_leaf_n':non_string,'example_n':len(records),'views':{}}
        for view,payload in [('strict_json_primary',False),('recognized_fenced_payload_diagnostic',True)]:
            correct={a:{} for a in ARMS}
            for row in rows:
                rec=recmap[row['id']]
                flags=prediction_leaves(rec,row,payload)
                correct[row['arm']].update({(row['id'],i,path):v for (i,path),v in flags.items()})
                if not payload:
                    verified=reviewed_score(rec,copy.deepcopy(row))
                    assert sum(flags.values())==verified['correct_args']
            split_out['views'][view]={
                'token_length':table(features,'token_bin',TOKEN_BINS,correct),
                'fragmentation':table(features,'fragment_bin',FRAGMENT_BINS,correct),
                'type':table(features,'type_bin',TYPE_BINS,correct)}
            for dimension in split_out['views'][view].values():
                assert sum(bin['string_leaf_n'] for bin in dimension.values())==len(features)
                for bin in dimension.values():
                    for p in bin['paired_attention_vs'].values():
                        assert p['wrong_to_correct']+p['correct_to_wrong']+p['both_correct']+p['both_wrong']==bin['string_leaf_n']
        output['splits'][split]=split_out
    dump(ROOT/'results/argument-subgroups.json',output)
    lines=['# Exploratory pilot argument subgroups','',
           '비공식 xLAM40/Hermes160 구현파일럿의 사후 기술 분석이다. 전체 공식 벤치마크나 확증적 endpoint가 아니다. 잘못된/누락 출력은 분모에 유지했다.',
           'Gold의 원래 unescaped 문자열과 고정 tokenizer로 문자·토큰·분할·유형을 계산했다. natural-language는 표면형 proxy다. 같은 함수명은 최대 정답 leaf 매칭을 사용하며 중복 호출의 그룹 귀속은 모호할 수 있다. N<20은 작은 표본으로 표시한다.','']
    for split,summary in output['splits'].items():
        lines += [f'## {split}: strings {summary["string_leaf_n"]}, non-string leaves {summary["non_string_leaf_n"]}','']
        for view,dimensions in summary['views'].items():
            for dimension,bins in dimensions.items():
                lines += [f'### {view} / {dimension}','','| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |',
                          '|---|---:|---:|---:|---:|---:|---:|---:|']
                for label,b in bins.items():
                    g=b['paired_attention_vs']['greedy'];m=b['paired_attention_vs']['match-forced']
                    lines.append(f'| {label}{" (small N)" if b["small_n"] else ""} | {b["string_leaf_n"]} | {b["example_n"]} | {b["correct"]["greedy"]} | {b["correct"]["match-forced"]} | {b["correct"]["attention+match-forced"]} | {g["wrong_to_correct"]}/{g["correct_to_wrong"]} | {m["wrong_to_correct"]}/{m["correct_to_wrong"]} |')
                lines.append('')
    lines += ['전체 5-arm correct/N, both correct/wrong, net change/N와 hash는 [small JSON](argument-subgroups.json)에 있다. Fenced-payload 진단은 primary strict JSON 점수를 대체하지 않는다. 원시 문자열과 개인정보는 이 요약에 포함하지 않았다.']
    (ROOT/'results/ARGUMENT_SUBGROUPS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({s:{k:v for k,v in d.items() if k!='views'} for s,d in output['splits'].items()}))


if __name__=='__main__':main()
