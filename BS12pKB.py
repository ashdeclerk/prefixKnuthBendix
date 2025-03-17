# This is an example file for prefixKnuthBendix.
# This file finds an autostackable structure for BS(1, 2).
# The original point of including BS(1, 2) was to show how to make a custom
# ordering, which `BS12pKB.py` does adequately. This is a more streamlined
# version, which I've included in the repo to demonstrate using the
# `piecewise_ordering` function from orderAutomata.
# This version appears to be slightly faster than the custom one, though I know
# not why. 

from prefixKnuthBendix.prefixKnuthBendix import *
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
file_handler = pkbLogging.make_file_handler("BS12.jsonl", level = 11, format_keys = format_keys)
stdout_handler = pkbLogging.make_stdout_handler(level = 19)
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
            # This is (language where u < v, language where v < u, language where u and v are incomparable)
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
orderings = [ordering_a, ordering_b, ordering_a, ordering_b, ordering_a, ordering_a]

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

def prune_prefixes(unresolved, rules, alph):
    # The naive way to prune prefixes gives us a lot of huge FSAs, which is
    # why I've kept it out of the naive implementation (because huge FSAs means
    # horrible slowdown). It's useful here for specific equations, though, so
    # we're using a less-naive implementation that prunes specifically prefixes 
    # for those equations 
    logger.log(major_steps, f"Pruning prefixes in {len(unresolved)} unresolved equations.")
    if len(rules) == 0:
        return True
    squared_alph = []
    for let1 in alph:
        for let2 in alph:
            squared_alph.append((let1, let2))
        squared_alph.append((let1, None))
        squared_alph.append((None, let1))
    partial_rewriter = complement(FSA.all_FSA(squared_alph))
    for rule in rules:
        words_dot_diag = singletons_diagonal_concatenate(rule.left, rule.right, alph)
        squared_prefixes = product(rule.prefixes, rule.prefixes)
        restricted_squared_prefixes = intersection(squared_prefixes, diagonal(alph))
        rule_rewriter = concatenation(restricted_squared_prefixes, words_dot_diag)
        partial_rewriter = union(partial_rewriter, rule_rewriter)
    bad_starts = (['T', 'a', 'a'], ['T', 'A', 'A'])
    for eqn in unresolved:
        if eqn.left[:3] in bad_starts or eqn.right[:3] in bad_starts:
            eqn.prefix_reduce(partial_rewriter)
    return True

def boundary_reduce(unresolved, rules, alph):
    # Same story here, but worse: Boundary reductions can easily lead to
    # infinite loops. Again, they're useful here for specific equations, so
    # we're using a non-naive implementation of boundary reduction.
    # Namely, we're only reducing at the boundary with free reduction. 
    logger.log(major_steps, f"Reducing at boundary in {len(unresolved)} unresolved equations.")
    checked = []
    bad_starts = (['T', 'a', 'a'], ['T', 'A', 'A'])
    while len(unresolved) > 0:
        eqn = unresolved.pop()
        if eqn.left[:3] in bad_starts or eqn.right[:3] in bad_starts:
            logger.log(handle_specific_equation, f"Reducing {eqn} at its boundary.")
            for rule in rules:
                if rule.left == ['t', 'T']:
                    possible_new_eqn = eqn.boundary_reduce(rule)
                    logger.log(handle_specific_equation, f"Added equation {possible_new_eqn}. Reduced original to {eqn}.")
                    if possible_new_eqn:
                        unresolved.append(possible_new_eqn)
        checked.append(eqn)
    for eqn in checked:
        unresolved.append(eqn)


pKB(BS12, max_time = 2000, max_rule_length = 100, prune_prefixes = prune_prefixes, boundary_reduce = boundary_reduce)

