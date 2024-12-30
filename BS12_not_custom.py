# This is an example file for prefixKnuthBendix.
# This file finds an autostackable structure for BS(1, 2).
# The original point of including BS(1, 2) was to show how to make a custom
# ordering, which `BS12pKB.py` does adequately. This is a more streamlined
# version, which I've included in the repo to demonstrate using the
# `piecewise_ordering` function from orderAutomata.
# This version appears to be slightly faster than the custom one, though I know
# not why. 

from prefixKnuthBendix.prefixKnuthBendix import Group, pKB
from prefixKnuthBendix.FSA import FSA
from prefixKnuthBendix.pkbLogging import pkbLogging
import atexit
import logging.config
import logging.handlers
from queue import SimpleQueue
import copy
from prefixKnuthBendix.orderAutomata.orderAutomata import piecewise_ordering

union_cache = {}
def union(fsa1, fsa2):
    frozenfsa1 = FSA.freeze(fsa1)
    frozenfsa2 = FSA.freeze(fsa2)
    if frozenfsa1 in union_cache:
        if frozenfsa2 in union_cache[frozenfsa1]:
            return union_cache[frozenfsa1][frozenfsa2]
    if frozenfsa2 in union_cache:
        if frozenfsa1 in union_cache[frozenfsa2]:
            return union_cache[frozenfsa2][frozenfsa1]
    if frozenfsa1 not in union_cache:
        union_cache[frozenfsa1] = {}
    fsa3 = FSA.union(fsa1, fsa2)
    union_cache[frozenfsa1][frozenfsa2] = fsa3
    return fsa3

logger = logging.getLogger("prefixKnuthBendix")
logger.setLevel(11)
format_keys = {
    "level": "levelname",
    "message": "message",
    "timestamp": "timestamp",
    "logger": "name",
    "module": "module",
    "function": "funcName",
    "line": "lineno",
    "thread_name": "threadName"
}
file_handler = pkbLogging.make_file_handler("BS12_copy.jsonl", level = 11, format_keys = format_keys)
stdout_handler = pkbLogging.make_stdout_handler(level = 13)
log_queue = SimpleQueue()
queue_handler = logging.handlers.QueueHandler(log_queue)
queue_listener = logging.handlers.QueueListener(log_queue, file_handler, stdout_handler, respect_handler_level = True)
queue_listener.start()
atexit.register(queue_listener.stop)
logger.addHandler(queue_handler)

BS12 = Group({'a', 't', 'A', 'T'}, [[['a','A'],[]], [['A', 'a'], []], [['t', 'T'], []], [['T', 't'], []], [['t', 'a', 'T'], ['a', 'a']], [['t', 'A', 'T'], ['A', 'A']]])
alph = {'a', 't', 'A', 'T'}


everything = FSA.all_FSA(alph)
nothing = FSA.empty_FSA(alph)
piecewise_automaton = FSA.FSA(6, set(), alph, {'a': [0, 1, 2, 4, 3, 5], 'A': [0, 1, 2, 4, 3, 5], 't': [5, 1, 2, 1, 2, 5], 'T': [3, 1, 2, 3, 3, 5]})
def ordering_a(u, v, L):
    scores_u = []
    scores_v = []
    for index, let in enumerate(u):
        if let == 'T':
            scores_u.append(u[:index].count('a') + u[:index].count('A'))
        elif let == 't':
            scores_u.append(u[index:].count('a') + u[index:].count('A') + 0.5)
    for index, let in enumerate(v):
        if let == 'T':
            scores_v.append(v[:index].count('a') + v[:index].count('A'))
        elif let == 't':
            scores_v.append(v[index:].count('a') + v[index:].count('A') + 0.5)
    for i in range(len(scores_u)):
        if scores_u[i] > scores_v[i]:
            return (nothing, everything, nothing)
        if scores_u[i] < scores_v[i]:
            return (everything, nothing, nothing)
    if len(u) > len(v):
        return (nothing, everything, nothing)
    if len(u) < len(v):
        return (everything, nothing, everything)
    return (nothing, nothing, everything)
def ordering_b(u, v, L):
    scores_u = []
    scores_v = []
    for index, let in enumerate(u):
        if let == 'T':
            scores_u.append(u[:index].count('a') + u[:index].count('A'))
        elif let == 't':
            scores_u.append(u[:index].count('a') + u[:index].count('A') + 0.5)
    for index, let in enumerate(v):
        if let == 'T':
            scores_v.append(v[:index].count('a') + v[:index].count('A'))
        elif let == 't':
            scores_v.append(v[:index].count('a') + v[:index].count('A') + 0.5)
    for i in range(len(scores_u)):
        if scores_u[i] > scores_v[i]:
            return (nothing, everything, nothing)
        if scores_u[i] < scores_v[i]:
            return (everything, nothing, nothing)
    if len(u) > len(v):
        return (nothing, everything, nothing)
    if len(u) < len(v):
        return (everything, nothing, everything)
    return (nothing, nothing, everything)
orderings = [ordering_a, ordering_b, ordering_a, ordering_a, ordering_a, ordering_a]

pieces = piecewise_ordering(piecewise_automaton, orderings)
# Note that the actual ordering is still a tie-break ordering. We need to do this
# specifically for the rules tT -> 1 and Tt -> 1, because they can misbehave
# by not ending at the same state. This is worth keeping in mind when you
# make your own regular piecewise partial order. 
def ordering(u, v, L):
    if u.count('t') + u.count('T') < v.count('t') + v.count('T'):
        return (everything, nothing, nothing)
    if u.count('t') + u.count('T') > v.count('t') + v.count('T'):
        return (nothing, everything, nothing)
    return pieces(u, v, L)

BS12.ordering = ordering
# We need to prune prefixes somewhat aggressively in this example. If we don't,
# then we end up keeping equations that look like a^n = Ta^(2n)t after
# any word that includes t. 
BS12.clean_first = True

pKB(BS12, max_time = 2000, max_rule_length = 100)

