# This is an example file for prefixKnuthBendix.
# This file finds an autostackable structure for BS(1, n)
# starting with n = 2 and increasing in value.
# Dealing with negative values of n was too annoying, so I've skipped it for now.
# I'm also not totally confident about what this was doing for n = 0 and n = 1,
# so those have been cut for now.

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
stdout_handler = pkbLogging.make_stdout_handler(level = 19)
file_handler = pkbLogging.make_file_handler(f"mass_BS1n.jsonl", level = 20, format_keys = format_keys)
log_queue = SimpleQueue()
queue_handler = logging.handlers.QueueHandler(log_queue)
queue_listener = logging.handlers.QueueListener(log_queue, file_handler, stdout_handler, respect_handler_level = True)
queue_listener.start()
atexit.register(queue_listener.stop)
logger.addHandler(queue_handler)

alph = {'a', 't', 'A', 'T'}
def BS1(n):
    return Group(alph, [[['a','A'],[]], [['A', 'a'], []], [['t', 'T'], []], [['T', 't'], []], [['t', 'a', 'T'], ['a'] * n], [['t', 'A', 'T'], ['A'] * n]])


everything = FSA.all_FSA(alph)
nothing = FSA.empty_FSA(alph)
# I hypothesize that you can readily adjust for BS(1, n) by changing the piecewise automaton as follows:
# The number of states should be 3 + n. States 1 and 2 are sinks, state 0 transitions via t to state 1 and via T to state 3,
# and states 3 through 2 + n form a cycle with a increasing amongst these states, A decreasing (both of those looping
# as needed), T going from any of these states to state 3, and t going to either state 2 (from state 3 only) or to state 1.
# Adjust the orderings appropriately -- states 2 and 3 get ordering b, but every other state gets ordering a. 
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

def pieces(n):
    transitions_a = [0, 1, 2] + list(range(4, 3 + n)) + [3]
    transitions_A = [0, 1, 2] + [2 + n] + list(range(3, 2 + n))
    piecewise_automaton = FSA.FSA(3 + n, set(), alph, {'a': transitions_a, 'A': transitions_A, 't': [1, 1, 2, 2] + [1] * (n - 1), 'T': [3, 1, 2] + [3] * n})
    orderings = [ordering_a, ordering_a, ordering_b, ordering_b] + [ordering_a] * (n - 1)
    return piecewise_ordering(piecewise_automaton, orderings)

# Note that the actual ordering is still a tie-break ordering. We need to do this
# specifically for the rules tT -> 1 and Tt -> 1, because they can misbehave
# by not ending at the same state. This is worth keeping in mind when you
# make your own regular piecewise partial order.
def generate_ordering(n):
    p = pieces(n)
    def ordering(u, v, L):
        if u.count('t') + u.count('T') < v.count('t') + v.count('T'):
            return (everything, nothing, nothing)
        if u.count('t') + u.count('T') > v.count('t') + v.count('T'):
            return (nothing, everything, nothing)
        return p(u, v, L)
    return ordering

def generate_prune(n):
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
            squared_alph.append((let1, ''))
            squared_alph.append(('', let1))
        partial_rewriter = complement(FSA.all_FSA(squared_alph))
        for rule in rules:
            words_dot_diag = singletons_diagonal_concatenate(rule.left, rule.right, alph)
            squared_prefixes = product(rule.prefixes, rule.prefixes)
            restricted_squared_prefixes = intersection(squared_prefixes, diagonal(alph))
            rule_rewriter = concatenation(restricted_squared_prefixes, words_dot_diag)
            partial_rewriter = union(partial_rewriter, rule_rewriter)
        bad_starts = (['T'] + ['a'] * n, ['T'] + ['A'] * n)
        for eqn in unresolved:
            if eqn.left[:n + 1] in bad_starts or eqn.right[:n + 1] in bad_starts:
                eqn.prefix_reduce(partial_rewriter)
        return True
    return prune_prefixes

def generate_boundary(n):
    def boundary_reduce(unresolved, rules, alph):
        logger.log(major_steps, f"Reducing at boundary in {len(unresolved)} unresolved equations.")
        checked = []
        bad_starts = (['T'] + ['a'] * n, ['T'] + ['A'] * n)
        while len(unresolved) > 0:
            eqn = unresolved.pop()
            if eqn.left[:n + 1] in bad_starts or eqn.right[:n + 1] in bad_starts:
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
    return boundary_reduce

n = 2
while True:
    logger.log(20, f"Starting BS(1, {n})")
    group = BS1(n)
    group.ordering = generate_ordering(n)
    pKB(group, max_time = 2000, prune_prefixes = generate_prune(n), boundary_reduce = generate_boundary(n))
    n += 1


