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

# We're using an alternate strategy here. It turns out that you need to
# both prune prefixes aggressively and do a bit of boundary reduction for this
# to work. I don't want that in the standard strategy.
def resolve_equations_alternate(unresolved, rules, ordering, int_pairs, ext_pairs, pre_pairs):
    # Let's see if delaying Ta^(2n)t = a^n equations helps
    new_unresolved = []
    while len(unresolved) > 0:
        current_equation = unresolved.pop()
        if (current_equation.left != [] and current_equation.right != []) and ((current_equation.left[0] == 'T' and current_equation.left[-1] == 't' and len(current_equation.left) > 2) or (current_equation.right[0] == 'T' and current_equation.right[-1] == 't' and len(current_equation.right) > 2)):
            new_unresolved.append(current_equation)
        else:
            logger.log(handle_specific_equation, f"Orienting equation: {current_equation}")
            possible_new_rules = current_equation.orient(ordering)
            if len(current_equation.prefixes.accepts) > 0:
                logger.log(equation_did_not_resolve, f"Returning unoriented equation {current_equation}")
                new_unresolved.append(current_equation)
            if len(possible_new_rules[0].prefixes.accepts) > 0:
                new_rule = possible_new_rules[0]
                logger.log(add_rule, f"Adding rule {new_rule}")
                for index, old_rule in enumerate(rules):
                    if len(new_rule.left) < len(old_rule.left):
                        int_pairs.append([len(rules), index])
                    elif len(new_rule.left) > len(old_rule.left):
                        int_pairs.append([index, len(rules)])
                    ext_pairs.append([index, len(rules)])
                    ext_pairs.append([len(rules), index])
                    pre_pairs.append([index, len(rules)])
                    pre_pairs.append([len(rules), index])
                rules.append(new_rule)
            if len(possible_new_rules[1].prefixes.accepts) > 0:
                new_rule = possible_new_rules[1]
                logger.log(add_rule, f"Adding rule {new_rule}")
                for index, old_rule in enumerate(rules):
                    if len(new_rule.left) < len(old_rule.left):
                        int_pairs.append([len(rules), index])
                    elif len(new_rule.left) > len(old_rule.left):
                        int_pairs.append([index, len(rules)])
                    ext_pairs.append([index, len(rules)])
                    ext_pairs.append([len(rules), index])
                    pre_pairs.append([index, len(rules)])
                    pre_pairs.append([len(rules), index])
                rules.append(new_rule)
    for eqn in new_unresolved:
        unresolved.append(eqn)
    return True

def pKB_alternate(group, max_rule_number = 1000, max_rule_length = None, max_time = 600):
    start_time = time.time()
    everything = FSA.all_FSA(group.generators)
    if hasattr(group, 'autostackableStructure'):
        if group.autostackableStructure.is_convergent:
            return None
        else:
            rules = group.autostackableStructure.rules
            int_pairs = group.autostackableStructure.int_pairs
            ext_pairs = group.autostackableStructure.ext_pairs
            pre_pairs = group.autostackableStructure.pre_pairs
            unresolved = group.autostackableStructure.unresolved
    else:
        rules = []
        int_pairs = []
        ext_pairs = []
        pre_pairs = []
        unresolved = []
        is_convergent = False
        for rel in group.relators:
            unresolved.append(Equation(rel[0], rel[1], copy.deepcopy(everything)))
    squared_alph = []
    for let1 in alph:
        for let2 in alph:
            squared_alph.append((let1, let2))
        squared_alph.append((let1, None))
        squared_alph.append((None, let1))
    while True:
        if len(int_pairs) > 0:
            logger.log(major_steps, f"Checking {len(int_pairs)} pairs of rules for interior critical pairs.")
            check_int_pairs(int_pairs, unresolved, group.generators, rules)
            clean_rules(rules, int_pairs, ext_pairs, pre_pairs)
        elif len(ext_pairs) > 0:
            logger.log(major_steps, f"Checking {len(ext_pairs)} pairs of rules for exterior critical pairs.")
            check_ext_pairs(ext_pairs, unresolved, group.generators, rules)
        elif len(pre_pairs) > 0:
            logger.log(major_steps, f"Checking {len(pre_pairs)} pairs of rules for prefix critical pairs.")
            check_pre_pairs(pre_pairs, unresolved, group.generators, everything, rules)
        logger.log(periodic_rule_display, f"Rules are {rules}")
        combine_equations(unresolved)
        rewrite_equations(unresolved, rules)
        logger.log(major_steps, f"Resolving {len(unresolved)} equations.")
        resolve_equations_alternate(unresolved, rules, group.ordering, int_pairs, ext_pairs, pre_pairs)
        if len(int_pairs) + len(ext_pairs) + len(pre_pairs) == 0 and len(unresolved) > 0:
            # We are going to prune prefixes selectively, because doing so
            # unselectively seems to be giving us huge ridiculous FSAs.
            if len(rules) > 0:
                partial_rewriter = complement(FSA.all_FSA(squared_alph))
                for rule in rules:
                    words_dot_diag = singletons_diagonal_concatenate(rule.left, rule.right, group.generators)
                    squared_prefixes = product(rule.prefixes, rule.prefixes)
                    restricted_squared_prefixes = intersection(squared_prefixes, diagonal(group.generators))
                    rule_rewriter = concatenation(restricted_squared_prefixes, words_dot_diag)
                    partial_rewriter = union(partial_rewriter, rule_rewriter)
            for eqn in unresolved:
                if eqn.left[:3] == ['T', 'a', 'a'] or eqn.left[:3] == ['T', 'A', 'A'] or eqn.right[:3] == ['T', 'a', 'a'] or eqn.right[:3] == ['T', 'A', 'A']:
                    eqn.prefix_reduce(partial_rewriter)
            # I don't want to rewrite *all* of the boundaries, just
            # the ones that I expect to give me trouble.
            # In this case, that's specifically Ta^(2n)t = a^n, and I only
            # want to freely reduce at the boundary, nothing more complex.
            checked = []
            while len(unresolved) > 0:
                eqn = unresolved.pop()
                if eqn.left[:3] == ['T', 'a', 'a'] or eqn.left[:3] == ['T', 'A', 'A'] or eqn.right[:3] == ['T', 'a', 'a'] or eqn.right[:3] == ['T', 'A', 'A']:
                    logger.log(13, f"Reducing {eqn} at its boundary!")
                    for rule in rules:
                        if rule.left == ['t', 'T']:
                            possible_new_eqn = eqn.boundary_reduce(rule)
                            logger.log(13, f"Added equation {possible_new_eqn}. Reduced original to {eqn}.")
                            if possible_new_eqn:
                                unresolved.append(possible_new_eqn)
                checked.append(eqn)
            for eqn in checked:
                unresolved.append(eqn)
            combine_equations(unresolved)
            rewrite_equations(unresolved, rules)
            resolve_equations(unresolved, rules, group.ordering, int_pairs, ext_pairs, pre_pairs)
        if len(unresolved) == 0:
            if len(int_pairs) + len(ext_pairs) + len(pre_pairs) == 0: # i.e., every equality has been resolved, after checking that there are no critical pairs left to check
                logger.log(logging.INFO, "Stopping now. Everything converges!")
                is_convergent = True
                break
        if len(rules) > max_rule_number:
            logger.log(logging.INFO, "Stopping now. There are too many rules!")
            break
        if time.time() - start_time > max_time:
            logger.log(logging.INFO, "Stopping now. This is taking too long!")
            break
        if check_rule_lengths(max_rule_length, unresolved):
            logger.log(logging.INFO, "Stopping now. The rules are getting too long!")
            break
    AS = AutostackableStructure(is_convergent, rules, int_pairs, ext_pairs, pre_pairs, unresolved)
    group.autostackableStructure = AS
    logger.log(logging.INFO, f"Total time taken: {time.time() - start_time} seconds")
    logger.log(logging.INFO, f"The current set of rules is {rules}")
    logger.log(logging.INFO, f"{'We successfully found an autostackable structure.' if is_convergent else 'We did not find an autostackable structure.'}")

pKB_alternate(BS12, max_time = 2000, max_rule_length = 100)

