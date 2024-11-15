# This is an example file for prefixKnuthBendix.
# This file finds an autostackable structure for BS(1, 2).
# You should look at Cox333pKB.py first. It's more thoroughly commented than this.

from prefixKnuthBendix.prefixKnuthBendix import Group, pKB
from prefixKnuthBendix.FSA import FSA
from prefixKnuthBendix.pkbLogging import pkbLogging
import atexit
import logging.config
import logging.handlers
from queue import SimpleQueue
import copy

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
stdout_handler = pkbLogging.make_stdout_handler(level = 13)
log_queue = SimpleQueue()
queue_handler = logging.handlers.QueueHandler(log_queue)
queue_listener = logging.handlers.QueueListener(log_queue, file_handler, stdout_handler, respect_handler_level = True)
queue_listener.start()
atexit.register(queue_listener.stop)
logger.addHandler(queue_handler)

# Note that you need to explicitly name the inverses and write down the
# inverse relations.
BS12 = Group({'a', 't', 'A', 'T'}, [[['a','A'],[]], [['A', 'a'], []], [['t', 'T'], []], [['T', 't'], []], [['t', 'a', 'T'], ['a', 'a']]])
alph = {'a', 't', 'A', 'T'}
expanded_alph = {'a', 't', 'A', 'T', None}
squared_alph = set()
for let in alph:
    for let2 in alph:
        squared_alph.add((let, let2))
    squared_alph.add((let, None))
    squared_alph.add((None, let))

# We're doing a bit of a weird ordering for this. It can be built from an FSA,
# but that version is incredibly slow.
# When you do this yourself, the key things to remember are:
# 1) The inputs of your ordering function are the words being compared
# *and* the language after which they're compared. You don't strictly need
# to do anything with the language, but you do need it as an input.
# For the remainder of this, I'll call the inputs v, w, and L (in that order).
# 2) The output of your ordering needs to be a triple (left, right, incomp),
# where left is an FSA accepting words u such that uv < uw,
# right is an FSA accepting words u such that uv > uw,
# and incomp is an FSA accepting all of the other words from inter. 

# If you're especially interested in this ordering in particular,
# we first count the number of ts and Ts (more is bigger), then use a piecewise
# ordering as a tiebreaker. The tiebreaker partitions all words into seven sets:
# i) Elements of {a, A}^2
# ii) Words with T before the first t and an even number of a/As before the first t but after the most recent T before that
# iii) Words with T before the first t and an odd number of a/As before the first t but after the most recent T before that
# iv) Words with T but no t and an even number of a/As after the last T
# v) Words with T but no t and an odd number of a/As after the last T
# vi) Words with t which do not have a T before the first t
# We only compare two words if they fall in the same set -- that's conveniently
# checked with finite state automata. If both words are in cases i, iii, iv, v, or vi
# we use ordering A. In case ii, we use ordering B.
# Both orderings A and B map words to N[1/2]*, i.e. lists of natural numbers possibly plus 1/2.
# We have an entry for each t and T, in the order they appear in the word.
# For both orderings, the entry for each T is the number of preceeding a/As.
# For ordering A, the entry for each t is the number of later a/As in the word + 1/2.
# For ordering B, the entry for each t is the number of earlier a/As in the word + 1/2.
# Splitting things into cases is mostly to get the ordering to play nice with
# concatenation on the right. The extra 1/2 is just to avoid T and t tying in
# an annoying way. Morally, ordering A is saying "move t to the right" and 
# ordering B is saying "move t to the left".
# Break ties with length. This is relevant because of aA.
# 
# I'm not 100% convinced that this *is* compatible with concatenation on the right
# in all cases. But it's late, and I want to get code running, so I'll deal with
# the proof later.  
# Update now that I've thought about it: Yeah, this is fine. Tedious to check,
# but all of the state orderings play nice with all of the transitions.
# (I.e., if u and v land at the same state and u < v with the ordering at that
# state, then ul < vl with the ordering at the state that ul and vl land at.)
# It's a straightforward induction proof from there. 

def ordering_a(u, v):
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
            return True
        elif scores_u[i] < scores_v[i]:
            return False
    return -1

def ordering_b(u, v):
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
            return True
        elif scores_u[i] < scores_v[i]:
            return False
    return -1

everything = FSA.all_FSA(alph)
nothing = FSA.empty_FSA(alph)
base_case_machine = FSA.FSA(6, {}, alph, {'a': [0, 1, 2, 4, 3, 5], 'A': [0, 1, 2, 4, 3, 5], 't': [5, 1, 2, 1, 2, 5], 'T': [3, 1, 2, 3, 3, 5]})
case_machines = [0] * 6
word_machines = [0] * 6
for i in range(6):
    case_machines[i] = copy.deepcopy(base_case_machine)
    case_machines[i].accepts = {i}
    word_machines[i] = copy.deepcopy(base_case_machine)
    word_machines[i].change_init(i) 
    # I'm very glad that I'm *not* putting things in BFS form when I create them or change their initial states

def ordering(u, v, L):
    # Easy case first. Different number of t-type letters means more is bigger.
    if u.count('t') + u.count('T') > v.count('t') + v.count('T'):
        return (nothing, everything, nothing)
    if u.count('t') + u.count('T') < v.count('t') + v.count('T'):
        return (everything, nothing, nothing)
    # Now we get to the big giant mess. u and v have the same number of t-type
    # letters, so we're in the piecewise part, and the piece we care about
    # depends on the prefix.
    # ~~This is going to be a huge mess of conditionals.~~ 
    # Scratch that, I figured out a much nicer way to do this.
    left = nothing
    right = nothing
    r = ordering_a(u, v)
    if r == -1:
        if len(u) > len(v):
            for prefix_case in range(6):
                final_case_u = word_machines[prefix_case].target_state(u)
                final_case_v = word_machines[prefix_case].target_state(v)
                if final_case_u == final_case_v:
                    if final_case_u in [0, 2, 3, 4, 5]:
                        right = union(right, case_machines[prefix_case])
        elif len(v) > len(u):
            for prefix_case in range(6):
                final_case_u = word_machines[prefix_case].target_state(u)
                final_case_v = word_machines[prefix_case].target_state(v)
                if final_case_u == final_case_v:
                    if final_case_u in [0, 2, 3, 4, 5]:
                        left = union(left, case_machines[prefix_case])
    elif r:
        for prefix_case in range(6):
            final_case_u = word_machines[prefix_case].target_state(u)
            final_case_v = word_machines[prefix_case].target_state(v)
            if final_case_u == final_case_v:
                if final_case_u in [0, 2, 3, 4, 5]:
                    right = union(right, case_machines[prefix_case])
    else:
        for prefix_case in range(6):
            final_case_u = word_machines[prefix_case].target_state(u)
            final_case_v = word_machines[prefix_case].target_state(v)
            if final_case_u == final_case_v:
                if final_case_u in [0, 2, 3, 4, 5]:
                    left = union(left, case_machines[prefix_case])
    r = ordering_b(u, v)
    if r == -1:
        if len(u) > len(v):
            for prefix_case in range(6):
                final_case_u = word_machines[prefix_case].target_state(u)
                final_case_v = word_machines[prefix_case].target_state(v)
                if final_case_u == final_case_v:
                    if final_case_u == 1:
                        right = union(right, case_machines[prefix_case])
        elif len(v) > len(u):
            for prefix_case in range(6):
                final_case_u = word_machines[prefix_case].target_state(u)
                final_case_v = word_machines[prefix_case].target_state(v)
                if final_case_u == final_case_v:
                    if final_case_u == 1:
                        left = union(left, case_machines[prefix_case])
    elif r:
        for prefix_case in range(6):
            final_case_u = word_machines[prefix_case].target_state(u)
            final_case_v = word_machines[prefix_case].target_state(v)
            if final_case_u == final_case_v:
                if final_case_u == 1:
                    right = union(right, case_machines[prefix_case])
    else:
        for prefix_case in range(6):
            final_case_u = word_machines[prefix_case].target_state(u)
            final_case_v = word_machines[prefix_case].target_state(v)
            if final_case_u == final_case_v:
                if final_case_u == 1:
                    left = union(left, case_machines[prefix_case])
    incomp = FSA.complement(union(left, right))
    return(left, right, incomp)

BS12.ordering = ordering

pKB(BS12, max_time = 2000, max_rule_length = 100)

