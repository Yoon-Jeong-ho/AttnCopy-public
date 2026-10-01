"""Order-independent leaf assignment matching the current RefCall definition.

This postprocessing correction leaves historical raw cached scores intact.
No source implementation is imported/copied; the assignment uses an independent
unit-capacity shortest-augmenting-path implementation.
"""
from .metrics import score, leaves, canonical
import json


def max_assignment(weights):
    n = len(weights)
    origin, destination = 2*n, 2*n+1
    graph = [[] for _ in range(2*n+2)]

    def edge(a,b,cost):
        forward = [b,1,cost,len(graph[b])]
        backward = [a,0,-cost,len(graph[a])]
        graph[a].append(forward)
        graph[b].append(backward)
        return forward

    choices = {}
    for i in range(n):
        edge(origin,i,0)
        edge(n+i,destination,0)
        for j in range(n):
            choices[i,j] = edge(i,n+j,-weights[i][j])
    for _ in range(n):
        distances, parents = [float('inf')]*len(graph), [None]*len(graph)
        distances[origin] = 0
        for _ in range(len(graph)-1):
            changed = False
            for node, links in enumerate(graph):
                for index, link in enumerate(links):
                    target, capacity, cost, _ = link
                    if capacity and distances[node]+cost < distances[target]:
                        distances[target] = distances[node]+cost
                        parents[target] = node,index
                        changed = True
            if not changed:
                break
        node = destination
        assert parents[node] is not None
        while node != origin:
            previous,index = parents[node]
            link = graph[previous][index]
            link[1] -= 1
            graph[node][link[3]][1] += 1
            node = previous
    return {i:j for (i,j),link in choices.items() if link[1] == 0}


def reviewed_score(record,result):
    s = score(record,result)
    if not s['structure_ok']:
        return s
    pred = json.loads(result['raw'])
    gold = record['answers']
    correct = 0
    for name in sorted({g['name'] for g in gold}):
        gs = sorted([g for g in gold if g['name']==name],key=lambda c:canonical([c]))
        ps = sorted([p for p in pred if p['name']==name],key=lambda c:canonical([c]))
        n = max(len(gs),len(ps))
        weight, counts = [[0]*n for _ in range(n)], {}
        for i,g in enumerate(gs):
            gl = dict(leaves(g['arguments']))
            for j,p in enumerate(ps):
                pl = dict(leaves(p['arguments']))
                count = sum(k in pl and type(pl[k]) is type(v) and pl[k]==v for k,v in gl.items())
                counts[i,j] = count
                weight[i][j] = count*(n+1)**2 + (canonical([g])==canonical([p]))*(n+1) + (set(g['arguments'])==set(p['arguments']))
        assignment = max_assignment(weight)
        correct += sum(counts.get((i,j),0) for i,j in assignment.items())
    s['correct_args'] = correct
    return s
