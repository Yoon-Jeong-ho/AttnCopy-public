"""Independent CPU postprocessing revision; preserves legacy scored predictions."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attncopy.bfcl_native import UPSTREAM, score_official
from attncopy.bfcl_scoring_review import REVISION, strict_arguments, native_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/bfcl-658"
ARMS=["greedy","attention-only","match-forced","attention+match-forced","verifiedPLD"]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return [json.loads(line) for line in Path(path).open()]
def ratio(a,b):return a/b if b else None
def main():
    assert json.loads((OUT/"evaluate/complete.json").read_text())["natural_exit"]
    prior=json.loads((ROOT/"results/bfcl-658.json").read_text())
    frozen=json.loads((OUT/"frozen.json").read_text())
    lineage=json.loads((OUT/"evaluate/lineage.json").read_text())
    assert subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"],text=True).strip()==prior["upstream_commit"]
    manifest=json.loads((ROOT/"docs/BFCL_SOURCE_MANIFEST.json").read_text())
    for name, info in manifest["artifacts"].items():
        assert digest(UPSTREAM/name)==info["sha256"],name
    for name, sha in lineage["code_sha256"].items():assert digest(ROOT/name)==sha,name
    assert digest(OUT/"frozen.json")==lineage["frozen_sha256"]
    assert digest(OUT/"runtime-inputs.jsonl")==frozen["runtime_sha256"]
    assert digest(OUT/"gold-strata.jsonl")==frozen["strata_sha256"]
    assert digest(OUT/"evaluate/predictions.jsonl")==prior["raw_predictions_sha256"]
    assert digest(OUT/"scoring/scored-predictions.jsonl")==prior["scored_predictions_sha256"]
    raw=read(OUT/"evaluate/predictions.jsonl")
    scored=read(OUT/"scoring/scored-predictions.jsonl")
    runtime=read(OUT/"runtime-inputs.jsonl")
    assert len(runtime)==658 and len(raw)==len(scored)==3290
    ids=[r["id"] for r in runtime]
    frozen_ids=[id for category in ["simple_python","live_simple"] for id in frozen["source_ids"][category]]
    assert ids==frozen_ids and len(set(ids))==658
    assert len({(r["id"],r["arm"]) for r in raw})==3290
    base=UPSTREAM/"berkeley-function-call-leaderboard/bfcl_eval/data"
    entries={}; gold={}
    for category,n in [("simple_python",400),("live_simple",258)]:
        e=read(base/("BFCL_v4_"+category+".json"));g=read(base/"possible_answer"/("BFCL_v4_"+category+".json"))
        assert len(e)==len(g)==n
        entries.update({r["id"]:r for r in e});gold.update({r["id"]:r for r in g})
    assert set(ids)==set(entries)==set(gold)
    directory=OUT/"postprocess-review-r1"
    directory.mkdir(exist_ok=False)
    reviewed=[]; differences=[]
    baseline={r["id"]:r for r in raw if r["arm"]=="greedy"}
    sidecar=directory/"reviewed-predictions.jsonl"
    with sidecar.open("w") as stream:
        for r,s in zip(raw,scored):
            assert (r["id"],r["arm"])==(s["id"],s["arm"])
            assert r["category"]==runtime[r["case_index"]]["category"]
            assert r["id"]==runtime[r["case_index"]]["id"]
            assert r["order"]==ARMS[r["case_index"]%5:]+ARMS[:r["case_index"]%5]
            assert r["prompt_tokens"]==len(runtime[r["case_index"]]["input_ids"])<frozen["context_limit"]
            assert r["generated_tokens"]==len(r["token_ids"])
            assert r["backend"]=="sdpa" and r["dtype"]=="bfloat16"
            check=score_official(entries[r["id"]],gold[r["id"]],r["raw"])
            assert check==s["official_scores"],(r["id"],r["arm"],"official score mismatch")
            strict=strict_arguments(entries[r["id"]],gold[r["id"]],check["parsed_calls"])
            proto=native_json(r["raw"])
            changed=[k for k,v in strict["per_argument"].items() if v is not None and v!=check["per_argument_correct"][k]]
            if changed:differences.append({"id":r["id"],"arm":r["arm"],"arguments":changed})
            identity=(r["token_ids"],r["stop"],r["eos_id"])==(baseline[r["id"]]["token_ids"],baseline[r["id"]]["stop"],baseline[r["id"]]["eos_id"])
            if r["arm"] in ["attention-only","verifiedPLD"]:assert identity
            row={"id":r["id"],"category":r["category"],"arm":r["arm"],"strict":strict,"protocol":proto,
                 "official_ast":check["official_ast_valid"],"parse_status":check["parse_status"],
                 "token_eos_identity":identity,"stop":r["stop"],"generated_tokens":r["generated_tokens"],
                 "raw_string_leaves_correct":sum(x["correct"] for x in s["string_flags"]),
                 "raw_string_leaves_n":len(s["string_flags"])}
            stream.write(json.dumps(row,ensure_ascii=False)+"\n");reviewed.append(row)
    tables={}
    for category,n in [("simple_python",400),("live_simple",258),("combined",658)]:
        tables[category]={}
        for arm in ARMS:
            rr=[r for r in reviewed if r["arm"]==arm and (category=="combined" or r["category"]==category)]
            assert len(rr)==n
            a=sum(r["strict"]["argument_n"] for r in rr);u=sum(r["strict"]["unknown_argument_n"] for r in rr)
            sa=sum(r["strict"]["string_argument_n"] for r in rr);su=sum(r["strict"]["unknown_string_argument_n"] for r in rr)
            ac=sum(r["strict"]["correct_argument_n"] for r in rr);sc=sum(r["strict"]["correct_string_argument_n"] for r in rr)
            cc=sum(r["strict"]["strict_call_correct_lower_bound"] for r in rr);ce=sum(r["strict"]["strict_call_evaluable"] for r in rr)
            table={"n":n,"official_ast_correct":sum(r["official_ast"] for r in rr),
                "strict_call_correct_lower_bound":cc,"strict_call_em_all_n_lower_bound":cc/n,
                "strict_call_evaluable_n":ce,"strict_call_unknown_n":n-ce,
                "argument_n":a,"undefined_gold_argument_n":u,"correct_exact_arguments":ac,
                "exact_argument_em_defined":ratio(ac,a-u),"exact_argument_em_all_n_lower_bound":ratio(ac,a),
                "top_level_string_argument_n":sa,"undefined_string_argument_n":su,"correct_raw_string_arguments":sc,
                "raw_string_argument_em_defined":ratio(sc,sa-su),
                "raw_string_leaf_n":sum(r["raw_string_leaves_n"] for r in rr),
                "raw_string_leaf_correct":sum(r["raw_string_leaves_correct"] for r in rr),
                "native_payload_json_valid_n":sum(r["protocol"]["native_payload_json_valid"] for r in rr),
                "native_fc_envelope_valid_n":sum(r["protocol"]["native_fc_envelope_valid"] for r in rr),
                "duplicate_json_key_n":sum(r["protocol"]["duplicate_json_key_n"] for r in rr),
                "native_protocol_status":dict(collections.Counter(r["protocol"]["status"] for r in rr)),
                "official_parse_status":dict(collections.Counter(r["parse_status"] for r in rr)),
                "stop_counts":dict(collections.Counter(r["stop"] for r in rr)),
                "token_eos_identity_n":sum(r["token_eos_identity"] for r in rr)}
            old=prior["tables"][category][arm]
            assert table["official_ast_correct"]==old["official_ast_correct"]
            assert table["raw_string_leaf_correct"]==old["correct_exact_string_leaves"]
            table["argument_count_delta_from_legacy"]=ac-old["correct_exact_arguments"]
            table["full_call_count_delta_from_legacy"]=cc-round(old["parsed_exact_typed_call_em"]*n)
            tables[category][arm]=table
    ids_table=[{"id":id,"category":runtime[i]["category"],"arms":{
         r["arm"]:{"ast":r["official_ast"],"strict_call_lower_bound":r["strict"]["strict_call_correct_lower_bound"],
                   "json":r["protocol"]["native_payload_json_valid"],"stop":r["stop"]} for r in reviewed if r["id"]==id}} for i,id in enumerate(ids)]
    summary={"revision":REVISION,"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "scope":"all simple_python400 + live_simple258, same frozen3290 predictions; CPU review only",
       "tables":tables,"changed_defined_argument_flags":differences,"all_case_ids":ids_table,
       "raw_sha256":digest(OUT/"evaluate/predictions.jsonl"),"legacy_scored_sha256":digest(OUT/"scoring/scored-predictions.jsonl"),
       "legacy_aggregate_sha256":digest(ROOT/"results/bfcl-658.json"),"reviewed_sidecar_sha256":digest(sidecar),
       "review_module_sha256":digest(ROOT/"attncopy/bfcl_scoring_review.py"),"review_script_sha256":digest(__file__),
       "generation_lineage":lineage,"checks":{"rows3290_unique":True,"all_category_ids":True,"official_ast_recomputed":True,
           "attention_and_verified_identity658":True,"source_and_frozen_generation_hashes_unchanged":True,"context_rejections":frozen["context_rejections"]},
       "definitions":["All accepted official alternatives; recursive exact types and strings; parsed escape differences only.",
          "7 empty-alternative arguments in2 cases undefined; defined-argument EM excludes them explicitly, whole-call all658 lower bound keeps2 unknown cases.",
          "Official AST and frozen string-leaf results remain unchanged; legacy shallow type flags are retained in the original sidecar.",
          "Native JSON validity requires at least one complete official-formatted final envelope and every payload valid JSON (no NaN/Infinity).",
          "Payload JSON syntax may be scalar; FC protocol additionally requires object/name-string/arguments-object and unambiguous duplicate-free keys.",
          "The shared official response cleaner removes reasoning before protocol checks. No call is not inferred to mean semantic refusal.",
          "This revision changes CPU metric semantics only, never generation, threshold, frozen features, official evaluator or runtime selection."]}
    dest=ROOT/"results/BFCL_VALIDATION.json";assert not dest.exists()
    dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (directory/"complete.json").write_text(json.dumps({"revision":REVISION,"rows":len(reviewed),"natural_exit":True})+"\n")
    print(json.dumps({"revision":REVISION,"changed_flags_n":len(differences),"tables":tables["combined"]}))
if __name__=="__main__":main()
