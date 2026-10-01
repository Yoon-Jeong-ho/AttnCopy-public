"""Fixed gold-side descriptive features; no output-dependent bin selection."""
import re

TOKEN_BINS=['0','1-4','5-16','17-64','>=65']
FRAGMENT_BINS=['empty','(0,0.25]','(0.25,0.5]','(0.5,1]','>1']
TYPE_BINS=['empty','URL','email','identifier','natural_language_proxy','other_string']


def string_features(value,token_count):
    n=len(value)
    token_bin='0' if token_count==0 else '1-4' if token_count<=4 else '5-16' if token_count<=16 else '17-64' if token_count<=64 else '>=65'
    ratio=token_count/n if n else None
    fragment='empty' if not n else '(0,0.25]' if ratio<=.25 else '(0.25,0.5]' if ratio<=.5 else '(0.5,1]' if ratio<=1 else '>1'
    if not value:kind='empty'
    elif re.fullmatch(r'(?:https?://|www\.)\S+',value):kind='URL'
    elif re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+',value):kind='email'
    elif re.fullmatch(r'[A-Za-z0-9_.:/#@+\-]+',value) and (any(c.isdigit() for c in value) or any(c in '_.:/#@+-' for c in value)):kind='identifier'
    elif any(c.isspace() for c in value):kind='natural_language_proxy'
    else:kind='other_string'
    return {'characters':n,'utf8_bytes':len(value.encode()),'token_count':token_count,'token_bin':token_bin,
            'fragmentation_ratio':ratio,'fragment_bin':fragment,'type_bin':kind}


def walk(obj,path=()):
    if isinstance(obj,dict):
        for k,v in obj.items():yield from walk(v,path+(k,))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):yield from walk(v,path+(i,))
    else:yield path,obj


def at_path(obj,path):
    for key in path:obj=obj[key]
    return obj


def official_gold_leaves(entry,gold):
    params=next(iter(gold['ground_truth'][0].values()))
    required=set(entry['function'][0]['parameters']['required'])
    result=[]
    for name,alternatives in params.items():
        if not alternatives:
            result.append({'path':[name],'accepted':[],'undefined_gold_reference':True,'optional_missing':False})
            continue
        reference=next((v for v in alternatives if v!=''),alternatives[0])
        for subpath,value in walk(reference):
            accepted=[]
            for alternative in alternatives:
                try:v=at_path(alternative,subpath)
                except (KeyError,IndexError,TypeError):continue
                if type(v) is type(value):accepted.append(v)
            result.append({'path':[name,*subpath],'reference':value,'accepted':accepted,
                           'optional_missing':name not in required and '' in alternatives})
    return result
