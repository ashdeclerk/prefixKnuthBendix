from .FSA import FSA
import copy
import time
import logging

logger = logging.getLogger("prefixKnuthBendix")
major_steps = 19
periodic_rule_display = 18
add_rule = 17
add_equation = 15
check_specific_pair = 13
remove_rule = 17
handle_specific_equation = 13
orient_specific_pair = 11
equation_did_not_resolve = 12
too_much_info = 9 # Do not use this unless you have a really simple group.


# Several caches to improve efficiency
# It turns out that calculating the same intersection a hundred times is slow.
intersection_cache = {}
def intersection(fsa1, fsa2):
    frozenfsa1 = FSA.freeze(fsa1)
    frozenfsa2 = FSA.freeze(fsa2)
    if frozenfsa1 in intersection_cache:
        if frozenfsa2 in intersection_cache[frozenfsa1]:
            return intersection_cache[frozenfsa1][frozenfsa2]
    if frozenfsa2 in intersection_cache:
        if frozenfsa1 in intersection_cache[frozenfsa2]:
            return intersection_cache[frozenfsa2][frozenfsa1]
    if frozenfsa1 not in intersection_cache:
        intersection_cache[frozenfsa1] = {}
    fsa3 = FSA.intersection(fsa1, fsa2)
    intersection_cache[frozenfsa1][frozenfsa2] = fsa3
    return fsa3

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

singleton_quotient_cache = {}
def singleton_quotient(fsa1, word):
    frozenfsa1 = FSA.freeze(fsa1)
    frozen_word = tuple(word)
    if frozenfsa1 in singleton_quotient_cache:
        if frozen_word in singleton_quotient_cache[frozenfsa1]:
            return singleton_quotient_cache[frozenfsa1][frozen_word]
    if frozenfsa1 not in singleton_quotient_cache:
        singleton_quotient_cache[frozenfsa1] = {}
    fsa3 = FSA.singleton_quotient(fsa1, word)
    singleton_quotient_cache[frozenfsa1][frozen_word] = fsa3
    return fsa3

BFS_cache = {}
def BFS(fsa1):
    frozenfsa1 = FSA.freeze(fsa1)
    if frozenfsa1 in BFS_cache:
        return BFS_cache[frozenfsa1]
    fsa2 = FSA.BFS(fsa1)
    BFS_cache[frozenfsa1] = fsa2
    return fsa2

diagonal_cache = {}
def diagonal(alphabet):
    frozenalph = frozenset(alphabet)
    if frozenalph in diagonal_cache:
        return diagonal_cache[frozenalph]
    fsa1 = FSA.diagonal(alphabet)
    diagonal_cache[frozenalph] = fsa1
    return fsa1

complement_cache = {}
def complement(fsa1):
    frozenfsa = FSA.freeze(fsa1)
    if frozenfsa in complement_cache:
        return complement_cache[frozenfsa]
    fsa2 = FSA.complement(fsa1)
    complement_cache[frozenfsa] = fsa2
    return fsa2

concatenation_cache = {}
def concatenation(fsa1, fsa2):
    frozenfsa1 = FSA.freeze(fsa1)
    frozenfsa2 = FSA.freeze(fsa2)
    if frozenfsa1 in concatenation_cache:
        if frozenfsa2 in concatenation_cache[frozenfsa1]:
            return concatenation_cache[frozenfsa1][frozenfsa2]
    if frozenfsa1 not in concatenation_cache:
        concatenation_cache[frozenfsa1] = {}
    fsa3 = FSA.concatenation(fsa1, fsa2)
    concatenation_cache[frozenfsa1][frozenfsa2] = fsa3
    return fsa3

single_word_FSA_cache = {}
def single_word_FSA(alphabet, word):
    alph = frozenset(alphabet)
    frword = tuple(word)
    if alph in single_word_FSA_cache:
        if frword in single_word_FSA_cache[alph]:
            return single_word_FSA_cache[alph][frword]
    if alph not in single_word_FSA_cache:
        single_word_FSA_cache[alph] = {}
    fsa = FSA.single_word_FSA(alphabet, word)
    single_word_FSA_cache[alph][frword] = fsa
    return fsa

product_cache = {}
def product(fsa1, fsa2):
    frozenfsa1 = FSA.freeze(fsa1)
    frozenfsa2 = FSA.freeze(fsa2)
    if frozenfsa1 in product_cache:
        if frozenfsa2 in product_cache[frozenfsa1]:
            return product_cache[frozenfsa1][frozenfsa2]
    if frozenfsa1 not in product_cache:
        product_cache[frozenfsa1] = {}
    fsa3 = FSA.product(fsa1, fsa2)
    product_cache[frozenfsa1][frozenfsa2] = fsa3
    return fsa3

projection_cache = {}
def projection(fsa1, indices):
    frozenfsa1 = FSA.freeze(fsa1)
    frozenindices = tuple(indices)
    if frozenfsa1 in projection_cache:
        if frozenindices in projection_cache[frozenfsa1]:
            return projection_cache[frozenfsa1][frozenindices]
    if frozenfsa1 not in projection_cache:
        projection_cache[frozenfsa1] = {}
    fsa2 = FSA.projection(fsa1, indices)
    projection_cache[frozenfsa1][frozenindices] = fsa2
    return fsa2

sing_diag_cache = {}
def singletons_diagonal_concatenate(word1, word2, alphabet):
    frozen_word1 = tuple(word1)
    frozen_word2 = tuple(word2)
    frozen_alph = frozenset(alphabet)
    if frozen_word1 in sing_diag_cache:
        if frozen_word2 in sing_diag_cache[frozen_word1]:
            if frozen_alph in sing_diag_cache[frozen_word1][frozen_word2]:
                return sing_diag_cache[frozen_word1][frozen_word2][frozen_alph]
            fsa = FSA.singletons_diagonal_concatenate(word1, word2, alphabet)
            frozen_word1[frozen_word2][frozen_alph] = fsa
            return fsa
        fsa = FSA.singletons_diagonal_concatenate(word1, word2, alphabet)
        sing_diag_cache[frozen_word1][frozen_word2] = {frozen_alph: fsa}
        return fsa
    fsa = FSA.singletons_diagonal_concatenate(word1, word2, alphabet)
    sing_diag_cache[frozen_word1] = {frozen_word2: {frozen_alph: fsa}}
    return fsa

def clear_caches():
    intersection_cache = {}
    singleton_quotient_cache = {}
    BFS_cache = {}
    diagonal_cache = {}
    complement_cache = {}
    concatenation_cache = {}
    single_word_FSA_cache = {}
    product_cache = {}
    projection_cache = {}
    sing_diag_cache = {}
    return None

class AutostackableStructure:

    def __init__(self, is_convergent, rules, int_pairs, ext_pairs, pre_pairs, unresolved):
        self.is_convergent = is_convergent
        self.rules = rules
        if is_convergent:
            pass
        else:
            self.int_pairs = int_pairs
            self.ext_pairs = ext_pairs
            self.pre_pairs = pre_pairs
            self.unresolved = unresolved

    def __strr__(self):
        out = f"Is convergent: {self.is_convergent}\n"
        out += "---\n"
        out += "Rules are: \n\n"
        for rule in self.rules:
            out += f"{rule}\n\n"
        if self.is_convergent:
            pass
        else:
            out += "---\n"
            out += "Interior Pairs are: \n"
            for pair in self.int_pairs:
                out += f"{pair}\n"
            out += "---\n"
            out += "Exterior Pairs are: \n"
            for pair in self.ext_pairs:
                out += f"{pair}\n"
            out += "---\n"
            out += "Prefix Pairs are: \n"
            for pair in self.pre_pairs:
                out += f"{pair}\n"
            out += "---\n"
            out += "Unresolved Equations are: \n"
            for pair in self.unresolved:
                out += f"{pair}\n"
        return out
        
class Group:

    def __init__(self, generators, relators):
        self.generators = generators
        self.relators = relators

class Rule:

    def __init__(self, left, right, prefixes):
        self.left = left
        self.right = right
        self.prefixes = prefixes
    
    def __repr__(self):
        return f"Rule({self.left}, {self.right}, {repr(self.prefixes)})"
    
    def __str__(self):
        return f"{self.left} => {self.right} after {self.prefixes}"

class Equation:

    def __init__(self, left, right, prefixes):
        self.left = copy.copy(left)
        self.right = copy.copy(right)
        self.prefixes = copy.deepcopy(prefixes)
    
    def __repr__(self):
        return f"Equation({self.left}, {self.right}, {repr(self.prefixes)})"
    
    def __str__(self):
        return f"{self.left} = {self.right} after {self.prefixes}"

    def reduce(self, rule):
        # Reduce either left or right using rule
        # Returns None if neither can be reduced using rule
        for index in range(len(self.left) - len(rule.left) + 1):
            if self.left[index:index + len(rule.left)] == rule.left:
                rewritable_prefixes = intersection(singleton_quotient(rule.prefixes, self.left[:index]), self.prefixes)
                new_left = self.left[:index] + rule.right + self.left[index + len(rule.left):]
                if len(rewritable_prefixes.accepts) > 0:
                    self.prefixes = BFS(intersection(self.prefixes, complement(rewritable_prefixes)))
                    if new_left != self.right:
                        return Equation(new_left, self.right, BFS(rewritable_prefixes))
        for index in range(len(self.right) - len(rule.left) + 1):
            if self.right[index:index + len(rule.left)] == rule.left:
                rewritable_prefixes = intersection(singleton_quotient(rule.prefixes, self.right[:index]), self.prefixes)
                new_right = self.right[:index] + rule.right + self.right[index + len(rule.left):]
                if len(rewritable_prefixes.accepts) > 0:
                    self.prefixes = BFS(intersection(self.prefixes, complement(rewritable_prefixes)))
                    if self.left != new_right:
                        return Equation(self.left, new_right, BFS(rewritable_prefixes))
        return None

    def prefix_reduce(self, rewriter):
        # Reduce prefixes using rule
        # We have to do this with annoying synchronous FSA stuff. Sorry.
        # This returns a bool, just so we can track "did something get reduced".
        # I'm not sure how helpful that is, but there might be situations where
        # somebody wants to use that as a "we're making progress" indicator.
        # We need to be a bit careful and reduce with all rules simultaneously,
        # otherwise we might add stuff back in that's already been removed.
        # TODO: That 
        L_cross_A_star = product(self.prefixes, FSA.all_FSA(self.prefixes.alphabet))
        synch = intersection(L_cross_A_star, rewriter)
        rewritten_words = complement(projection(synch, [0]))
        if len(rewritten_words.accepts) > 0:
            new_prefixes = intersection(union(self.prefixes, projection(synch, [1])),
                                        complement(projection(rewriter, [0])))
            self.prefixes = copy.deepcopy(new_prefixes)
            return True
        return False

    def boundary_reduce(self, rule):
        # Reduce at the boundary between prefixes and either left or right
        # This is slightly dangerous, because it ends up replacing equations by 
        # more complex equations in some situations. It's potentially useful, so
        # I've written the code for it, but I haven't included it in the base
        # strategy for pKB.
        # Apply sparingly, is the punchline. Or don't. I'm not your mom. 
        for index in range(1, min(len(rule.left), len(self.left))):
            if self.left[:index] == rule.left[-index:]:
                w1 = rule.left[:-index]
                w3 = self.left[index:]
                rewritten_prefixes = intersection(rule.prefixes, singleton_quotient(self.prefixes, w1))
                if len(rewritten_prefixes.accepts) > 0:
                    self.prefixes = BFS(intersection(self.prefixes, complement(concatenation(rule.prefixes, single_word_FSA(self.prefixes.alphabet, w1)))))
                    return Equation(rule.right + w3, w1 + self.right, BFS(rewritten_prefixes))
        for index in range(1, min(len(rule.left), len(self.right))):
            if self.right[:index] == rule.left[-index:]:
                w1 = rule.left[:-index]
                w3 = self.right[index:]
                rewritten_prefixes = intersection(rule.prefixes, singleton_quotient(self.prefixes, w1))
                if len(rewritten_prefixes.accepts) > 0:
                    self.prefixes = BFS(intersection(self.prefixes, complement(concatenation(rule.prefixes, single_word_FSA(self.prefixes.alphabet, w1)))))
                    return Equation(rule.right + w3, w1 + self.right, BFS(rewritten_prefixes))
        return None

    def orient(self, order):
        # Orient where possible using order
        # Returns a pair of rules. You'll need to do some post-processing
        # in case self.prefixes or either set of rule prefixes is empty.
        (left_better, right_better, incomp) = order(self.left, self.right, self.prefixes)
        self.prefixes = BFS(incomp)
        return (Rule(self.left, self.right, BFS(right_better)), Rule(self.right, self.left, BFS(left_better)))

class RewritingChain:

    def __init__(self, words):
        self.words = words
    
    def __repr__(self):
        return f"RewritingChain({self.words})"
    
    def __str__(self):
        return f'{" => ".join(str(word) for word in self.words)}'


def check_int_pairs(int_pairs, unresolved, alphabet, rules):
    logger.log(11, f"Rules are {rules}")
    while len(int_pairs) > 0:
        pair = int_pairs.pop()
        pair[0] = rules[pair[0]]
        pair[1] = rules[pair[1]]
        logger.log(check_specific_pair, f"Checking for interior critical pairs between {pair[0]} and {pair[1]}")
        for index in range(0, len(pair[1].left) - len(pair[0].left) + 1):
            if pair[0].left == pair[1].left[index: index + len(pair[0].left)]:
                word1 = copy.copy(pair[1].right)
                word2 = copy.copy(pair[1].left)
                word2[index:index + len(pair[0].left)] = pair[0].right
                prefixes = intersection(pair[1].prefixes, singleton_quotient(pair[0].prefixes, pair[1].left[0:index]))
                while len(word1) > 0 and len(word2) > 0 and word1[-1] == word2[-1]:
                    del word1[-1]
                    del word2[-1]
                old_prefixes = copy.deepcopy(prefixes)
                p = []
                while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                    p.append(word1.pop(0))
                    del word2[0]
                if len(p) > 0:
                    prefixes = concatenation(prefixes, single_word_FSA(prefixes.alphabet, p))
                if len(prefixes.accepts) > 0 and word1 != word2:
                    new_equation = Equation(word1, word2, copy.deepcopy(prefixes))
                    logger.log(add_equation, f"Adding equation {new_equation}")
                    unresolved.append(new_equation)
                    # We need to remove redundant portions of rules. If we don't,
                    # then there seems to be an issue where equations don't fully
                    # reduce for some reason. I've no clue how that's happening,
                    # because it shouldn't, but this also makes things a bit more
                    # efficient because we have fewer rules to fuss with. 
                    logger.log(remove_rule, f"Removing redundant portion of {pair[1]}")
                    pair[1].prefixes = intersection(pair[1].prefixes, complement(old_prefixes))
                elif len(prefixes.accepts) > 0:
                    # This piece of garbage case has been annoying me for like
                    # two hours at time of writing, because I didn't think
                    # that it would ever happen. In this case, our critical pair
                    # resolves itself, and we can just cut appropriate parts
                    # of the longer rule.
                    logger.log(remove_rule, f"Removing redundant portion of {pair[1]}")
                    pair[1].prefixes = intersection(pair[1].prefixes, complement(old_prefixes))
    return True 

def clean_rules(rules, int_pairs, ext_pairs, pre_pairs):
    removable_rules = []
    for index, rule in enumerate(rules):
        if len(rule.prefixes.accepts) == 0:
            removable_rules.append(index)
    while len(removable_rules) > 0:
        rule_index = removable_rules.pop()
        logger.log(remove_rule, f"Removing empty rule {rules.pop(rule_index)}")
        removable_pairs = []
        for index, pair in enumerate(int_pairs):
            if pair[0] == rule_index or pair[1] == rule_index:
                removable_pairs.append(index)
            if pair[0] > rule_index:
                pair[0] -= 1
            if pair[1] > rule_index:
                pair[1] -= 1
        while len(removable_pairs) > 0:
            int_pairs.pop(removable_pairs.pop())
        removable_pairs = []
        for index, pair in enumerate(ext_pairs):
            if pair[0] == rule_index or pair[1] == rule_index:
                removable_pairs.append(index)
            if pair[0] > rule_index:
                pair[0] -= 1
            if pair[1] > rule_index:
                pair[1] -= 1
        while len(removable_pairs) > 0:
            ext_pairs.pop(removable_pairs.pop())
        removable_pairs = []
        for index, pair in enumerate(pre_pairs):
            if pair[0] == rule_index or pair[1] == rule_index:
                removable_pairs.append(index)
            if pair[0] > rule_index:
                pair[0] -= 1
            if pair[1] > rule_index:
                pair[1] -= 1
        while len(removable_pairs) > 0:
            pre_pairs.pop(removable_pairs.pop())
    return True

def check_ext_pairs(ext_pairs, unresolved, alphabet, rules):
    while len(ext_pairs) > 0:
        pair = ext_pairs.pop()
        pair[0] = rules[pair[0]]
        pair[1] = rules[pair[1]]
        logger.log(check_specific_pair, f"Checking for exterior critical pairs between {pair[0]} and {pair[1]}")
        # We're looking for a proper suffix of l1 to be a prefix of l2
        for index in range(1, len(pair[0].left)):
            if pair[0].left[index:] == pair[1].left[:len(pair[0].left) - index]:
                word1 = pair[0].left[:index] + pair[1].right
                word2 = pair[0].right + pair[1].left[len(pair[0].left) - index:]
                # Prefix language is (L2 / pair[0][1][:index]) \cap L1
                prefixes = copy.deepcopy(pair[1].prefixes)
                prefixes = singleton_quotient(prefixes, pair[0].left[:index])
                prefixes = intersection(prefixes, pair[0].prefixes)
                prefixes = BFS(prefixes)
                if word1 != word2 and len(prefixes.accepts) > 0:
                    p = []
                    while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                        p.append(word1.pop(0))
                        del word2[0]
                    if len(p) > 0:
                        prefixes = concatenation(prefixes, single_word_FSA(prefixes.alphabet, p))
                    if len(prefixes.accepts) > 0 and word1 != word2:
                        new_equation = Equation(word1, word2, copy.deepcopy(prefixes))
                        logger.log(add_equation, f"Adding equation {new_equation}")
                        unresolved.append(new_equation)
    return True

def check_pre_pairs(pre_pairs, unresolved, alphabet, everything, rules):
    while len(pre_pairs) > 0:
        pair = pre_pairs.pop()
        pair[0] = rules[pair[0]]
        pair[1] = rules[pair[1]]
        logger.log(check_specific_pair, f"Checking for prefix critical pairs between {pair[0]} and {pair[1]}")
        prefixes = copy.deepcopy(pair[0].prefixes)
        prefixes = product(prefixes, prefixes)
        prefixes = intersection(prefixes, diagonal(alphabet))
        prefixes = concatenation(prefixes, FSA.singletons_diagonal_concatenate(pair[0].left, pair[0].right, alphabet))
        bad_prefixes = projection(intersection(prefixes, product(pair[1].prefixes, complement(pair[1].prefixes))), [1])
        if len(bad_prefixes.accepts) > 0:
            word1 = copy.copy(pair[1].left)
            word2 = copy.copy(pair[1].right)
            p = []
            while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                p.append(word1.pop(0))
                del word2[0]
            if len(p) > 0:
                bad_prefixes = concatenation(bad_prefixes, single_word_FSA(prefixes.alphabet, p))
            if len(prefixes.accepts) > 0 and word1 != word2:
                new_equation = Equation(word1, word2, copy.deepcopy(bad_prefixes))
                logger.log(add_equation, f"Adding equation {new_equation}")
                unresolved.append(new_equation)
        # This removes "irrelevant" prefixes, i.e. stuff that should get
        # rewritten by pair[0]. I'm not sure it's strictly necessary, but
        # it might help in some situations.
        # It does slow things way the heck down, because it complicates the FSAs
        # we're working with. So it's commented out for the moment. 
        #pair[1].prefixes = intersection(pair[1].prefixes, complement(projection(prefixes, [0])))
    return True

def rewrite_equations(unresolved, rules):
    fully_reduced = []
    while len(unresolved) > 0:
        current_equation = unresolved.pop()
        logger.log(handle_specific_equation, f"Attempting to reduce {current_equation}")
        reduced = False
        for rule in rules:
            old_equation = copy.deepcopy(current_equation)
            possible_new_equation = current_equation.reduce(rule)
            if possible_new_equation:
                unresolved.append(possible_new_equation)
                logger.log(too_much_info, f"Reduced {old_equation} to {possible_new_equation} using {rule}")
                reduced = True
        if reduced and len(current_equation.prefixes.accepts) > 0:
            # We might need to go through this equation again because it can be
            # reduced by a rule in multiple spots. Each pass of `reduce` only
            # reduces in one spot. 
            unresolved.append(current_equation)
        elif len(current_equation.prefixes.accepts) > 0:
            fully_reduced.append(current_equation)
        else:
            logger.log(too_much_info, f"Forgetting empty equation {current_equation}")
    for eqn in fully_reduced:
        unresolved.append(eqn)
    return True

def resolve_equations(unresolved, rules, ordering, int_pairs, ext_pairs, pre_pairs):
    new_unresolved = []
    while len(unresolved) > 0:
        current_equation = unresolved.pop()
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

def combine_equations(unresolved):
    for i in range(len(unresolved) - 1):
        if len(unresolved[i].prefixes.accepts) > 0:
            for j in range(i + 1, len(unresolved)):
                if unresolved[i].left == unresolved[j].left and unresolved[i].right == unresolved[j].right:
                    logger.log(handle_specific_equation, f"Merging equation {unresolved[j]} into {unresolved[i]}")
                    unresolved[i].prefixes = union(unresolved[i].prefixes, unresolved[j].prefixes)
                    unresolved[j].prefixes.accepts = set()
                elif unresolved[i].left == unresolved[j].right and unresolved[i].right == unresolved[j].left:
                    logger.log(handle_specific_equation, f"Merging equation {unresolved[j]} into {unresolved[i]}")
                    unresolved[i].prefixes = union(unresolved[i].prefixes, unresolved[j].prefixes)
                    unresolved[j].prefixes.accepts = set()
    for i in range(len(unresolved) - 1, -1, -1):
        if len(unresolved[i].prefixes.accepts) == 0:
            logger.log(handle_specific_equation, f"Removing empty equation {unresolved[i]}")
            del unresolved[i]
    return True

def prune_prefixes_naive(unresolved, rules, alph):
    # This gives us huges FSAs when called, which means I've cut it from
    # the default implementation of pKB. If you want to use it, pass it
    # when you call pKB. You can also define your own function to prune prefixes,
    # as I've done in the BS(1, 2) example file. 
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
    for eqn in unresolved:
        eqn.prefix_reduce(partial_rewriter)
    return True

def check_rule_lengths(max_rule_length, unresolved):
    # Checks for any rules that are too long
    if max_rule_length == None:
        return False
    for equality in unresolved:
        if len(equality.left) > max_rule_length:
            return True
        if len(equality.right) > max_rule_length:
            return True

def autostackableNormalForms(group):
    # Gives the normal forms of a group with an autostackable structure.
    if not hasattr(group, 'autostackableStructure'):
        return None
    if not group.autostackableStructure.is_convergent:
        return None
    ev = FSA.all_FSA(group.generators)
    nf = FSA.all_FSA(group.generators)
    for rule in group.autostackableStructure.rules:
        # Useful note: If we swap the order in the next line, we have a race condition. Presumably because the computer grabs nf before it can be rewritten.
        nf = intersection(complement(concatenation(concatenation(rule.prefixes, single_word_FSA(group.generators, rule.left)), ev)), nf)
    return nf

def pKB(group, max_rule_number = 1000, max_rule_length = None, max_time = 600, prune_prefixes = lambda unresolved, rules, gens : None, boundary_reduce = lambda unresolved, rules, gens : None):
    # We'll assume that the ordering is stored in group.ordering. I trust the user, because the user is likely me at this point.
    # We should also check if there's partial progress on a rewriting system.
    # Initial step:
    start_time = time.time()
    everything = FSA.all_FSA(group.generators)
    if hasattr(group, 'autostackableStructure'):
        if group.autostackableStructure.is_convergent:
            return None # We already have an autostackable structure; we're done
        else: # We have a partial but incomplete solution.
            rules = group.autostackableStructure.rules
            # It may be worth tossing rules into unresolved?
            # No, that's slightly silly. On a genuine ordering change, user should also clean the group's autostackable structure.
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
    # Initial step complete. Now we loop between the (critpair check/equality resolution) loop and the (check unresolved/prefix reduction) loop.
    while True:
        # This type of loop (where we continue forever and break at a certain condition)
        # is apparently called a "loop and a half"; python, unfortunately, lacks 
        # a native "until" loop type.
        # Check for critical pairs
        # Interior as top priority, just because they are quick to check and eliminate things
        # Exterior as a second priority, i.e. todo if there are no pairs that we haven't check for interior pairs. Harder to check, but not as bad as:
        # Prefix as a bottom priority. THese are *slow* to check, since it's all about constructing FSAs.
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
        # Equality resolution
        logger.log(periodic_rule_display, f"Rules are {rules}")
        if len(unresolved) > 0:
            prune_prefixes(unresolved, rules, group.generators)
            boundary_reduce(unresolved, rules, group.generators)
            combine_equations(unresolved)
            rewrite_equations(unresolved, rules)
        logger.log(major_steps, f"Resolving {len(unresolved)} equations.")
        resolve_equations(unresolved, rules, group.ordering, int_pairs, ext_pairs, pre_pairs)
        # And now we check if we need to halt.
        if len(unresolved) == 0:
            if len(int_pairs) + len(ext_pairs) + len(pre_pairs) == 0: # i.e., every equality has been resolved, after checking that there are no critical pairs left to check
                logger.log(logging.INFO, "Stopping now. Everything converges!")
                is_convergent = True
                break
        # If we have too many rules
        if len(rules) > max_rule_number:
            logger.log(logging.INFO, "Stopping now. There are too many rules!")
            break
        # If we've spent too much time
        if time.time() - start_time > max_time:
            logger.log(logging.INFO, "Stopping now. This is taking too long!")
            break
        # If equalities are too long to compare
        if check_rule_lengths(max_rule_length, unresolved):
            logger.log(logging.INFO, "Stopping now. The rules are getting too long!")
            break
    # Okay, we've stopped. Now we need to update group appropriately.
    AS = AutostackableStructure(is_convergent, rules, int_pairs, ext_pairs, pre_pairs, unresolved)
    group.autostackableStructure = AS
    logger.log(logging.INFO, f"Total time taken: {time.time() - start_time} seconds")
    logger.log(logging.INFO, f"The current set of rules is {rules}")
    logger.log(logging.INFO, f"{'We successfully found an autostackable structure.' if is_convergent else 'We did not find an autostackable structure.'}")
