"""Official Qwen-FC CPU prompt/parser/scorer adapter, no API execution."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'data/bfcl-upstream-6ea5797'
sys.path.insert(0,str(UPSTREAM/'berkeley-function-call-leaderboard'))
REGISTRY='Qwen/Qwen3-0.6B-FC'


def handler():
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    config=MODEL_CONFIG_MAPPING[REGISTRY]
    assert config.model_handler is QwenFCHandler and config.is_fc_model and not config.underscore_to_dot
    # Pure formatting/parsing methods do not need the server/client constructor.
    return QwenFCHandler.__new__(QwenFCHandler)


def native_prompt(entry):
    h=handler()
    original=copy.deepcopy(entry)
    assert len(entry['question'])==1
    data=h._pre_query_processing_prompting(copy.deepcopy(entry))
    data=h.add_first_turn_message_prompting(data,copy.deepcopy(entry['question'][0]))
    prompt=h._format_prompt(data['message'],data['function'])
    marked=copy.deepcopy(data['message'])
    source_items=[]
    for i,message in enumerate(marked):
        if message['role']=='user' and isinstance(message['content'],str):
            marker=f'␀BFCL_SOURCE_{i}␀'
            assert marker not in prompt
            source_items.append((marker,message['content'],i))
            message['content']=marker
    template=h._format_prompt(marked,data['function'])
    sources=[]
    for marker,text,i in source_items:
        start=template.index(marker)
        assert template.count(marker)==1
        template=template.replace(marker,text,1)
        sources.append({'text':text,'message_index':i,'prompt_char_start':start})
    assert template==prompt and entry==original
    return prompt,sources


def exact_value(value,alternatives):
    return any(type(value) is type(g) and value==g for g in alternatives)


def score_official(entry,gold,raw):
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language
    h=handler()
    status='parsed'
    try:
        # Follow the handler's official response cleaning before AST decode;
        # reasoning examples must not be treated as final tool calls.
        response=SimpleNamespace(choices=[SimpleNamespace(text=raw)],
                  usage=SimpleNamespace(prompt_tokens=0,completion_tokens=0))
        cleaned=h._parse_query_response_prompting(response)
        parsed=h.decode_ast(cleaned['model_responses'],Language.PYTHON,True)
    except Exception as error:
        parsed=[];status='parser_exception'
        parser_error=type(error).__name__+': '+str(error)
    try:
        ast=ast_checker(copy.deepcopy(entry['function']),copy.deepcopy(parsed),copy.deepcopy(gold['ground_truth']),Language.PYTHON,
                        'simple_python' if entry['id'].startswith('simple_python_') else 'live_simple',REGISTRY)
    except Exception as error:
        ast={'valid':False,'error_type':'ast_exception','error':[type(error).__name__+': '+str(error)]}
        if status=='parsed':status='ast_exception'
    if not parsed and status=='parsed':
        status='no_parsed_call'
    possible=gold['ground_truth'][0]
    name=next(iter(possible))
    expected=possible[name]
    required=set(entry['function'][0]['parameters']['required'])
    has_call=len(parsed)==1 and isinstance(parsed[0],dict) and name in parsed[0] and isinstance(parsed[0][name],dict)
    pred=parsed[0][name] if has_call else {}
    correct={k:has_call and ((k in pred and exact_value(pred[k],alts)) or (k not in pred and k not in required and '' in alts))
             for k,alts in expected.items()}
    strict_call=(has_call and required<=set(pred) and set(pred)<=set(expected) and all(correct.values()))
    string_keys=[k for k,alts in expected.items() if any(isinstance(v,str) and v!='' for v in alts)
                 or entry['function'][0]['parameters']['properties'].get(k,{}).get('type')=='string']
    return {'official_ast_valid':bool(ast['valid']),'official_ast':ast,'parse_status':status,'parsed_calls':parsed,
            'official_response_cleaner':'QwenFCHandler._parse_query_response_prompting',
            'parsed_raw_value_call_em':strict_call,'gold_argument_n':len(expected),'correct_arguments':sum(correct.values()),
            'string_argument_n':len(string_keys),'correct_raw_string_arguments':sum(correct[k] for k in string_keys),
            'per_argument_correct':correct,'gold_alternatives_policy':'all official alternatives, exact parsed values/types; optional empty sentinel permits omission',
            'raw_string_normalization':'none; escaping removed only by JSON parsing'}
