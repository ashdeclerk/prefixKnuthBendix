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

quotient_cache = {}
def quotient(fsa1, fsa2):
    frozenfsa1 = FSA.freeze(fsa1)
    frozenfsa2 = FSA.freeze(fsa2)
    if frozenfsa1 in quotient_cache:
        if frozenfsa2 in quotient_cache[frozenfsa1]:
            return quotient_cache[frozenfsa1][frozenfsa2]
    if frozenfsa1 not in quotient_cache:
        quotient_cache[frozenfsa1] = {}
    fsa3 = FSA.quotient(fsa1, fsa2)
    quotient_cache[frozenfsa1][frozenfsa2] = fsa3
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

def clear_caches():
    intersection_cache = {}
    quotient_cache = {}
    BFS_cache = {}
    diagonal_cache = {}
    complement_cache = {}
    concatenation_cache = {}
    single_word_FSA_cache = {}
    product_cache = {}
    projection_cache = {}
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

    def __repr__(self):
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


def check_int_pairs(int_pairs, ext_pairs, pre_pairs, unresolved, alphabet, rules):
    while len(int_pairs) > 0:
        pair = int_pairs.pop()
        pair[0] = rules[pair[0]]
        pair[1] = rules[pair[1]]
        logger.log(check_specific_pair, f"Checking for interior critical pairs between {pair[0]} and {pair[1]}")
        for index in range(0, len(pair[1][1]) - len(pair[0][1]) + 1):
            if pair[0][1] == pair[1][1][index: index + len(pair[0][1])]:
                word1 = copy.copy(pair[1][2])
                word2 = copy.copy(pair[1][1])
                word2[index:index + len(pair[0][1])] = pair[0][2]
                prefixes = copy.deepcopy(pair[1][0])
                prefixes = concatenation(prefixes, single_word_FSA(alphabet, pair[1][1][0:index]))
                prefixes = intersection(prefixes, pair[0][0])
                prefixes = quotient(prefixes, single_word_FSA(alphabet, pair[1][1][0:index]))
                prefixes = BFS(prefixes)
                while len(word1) > 0 and len(word2) > 0 and word1[-1] == word2[-1]:
                    del word1[-1]
                    del word2[-1]
                p = []
                while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                    p.append(word1.pop(0))
                    del word2[0]
                rawPrefixes = copy.deepcopy(prefixes)
                if len(p) > 0:
                    prefixes = concatenation(prefixes, single_word_FSA(prefixes.alphabet, p))
                if len(prefixes.accepts) > 0 and word1 != word2:
                    logger.log(add_equation, f"Adding equation {word1} = {word2} after {prefixes}")
                    unresolved.append([prefixes, word1, word2])
                deleted_rule_indices = []
                for jndex in range(0, len(rules)):
                    if rules[jndex][1] == pair[1][1]:
                        if rules[jndex][2] == pair[1][2]:
                            rules[jndex][0] = BFS(intersection(rules[jndex][0], complement(rawPrefixes)))
                            if len(rules[jndex][0].accepts) == 0:
                                deleted_rule_indices.append(jndex)
                while len(deleted_rule_indices) > 0:
                    jndex = deleted_rule_indices.pop(-1)
                    logger.log(remove_rule, f"Removing redundant rule {rules[jndex]}")
                    del rules[jndex]
                    for k in list(range(len(int_pairs) - 1, -1, -1)):
                        if int_pairs[k][0] == jndex or int_pairs[k][1] == jndex:
                            del int_pairs[k]
                        else:
                            if int_pairs[k][0] > jndex:
                                int_pairs[k][0] -= 1
                            if int_pairs[k][1] > jndex:
                                int_pairs[k][1] -= 1
                    for k in list(range(len(ext_pairs) - 1, -1, -1)):
                        if ext_pairs[k][0] == jndex or ext_pairs[k][1] == jndex:
                            del ext_pairs[k]
                        else:
                            if ext_pairs[k][0] > jndex:
                                ext_pairs[k][0] -= 1
                            if ext_pairs[k][1] > jndex:
                                ext_pairs[k][1] -= 1
                    for k in list(range(len(pre_pairs) - 1, -1, -1)):
                        if pre_pairs[k][0] == jndex or pre_pairs[k][1] == jndex:
                            del pre_pairs[k]
                        else:
                            if pre_pairs[k][0] > jndex:
                                pre_pairs[k][0] -= 1
                            if pre_pairs[k][1] > jndex:
                                pre_pairs[k][1] -= 1
                
                

def check_ext_pairs(ext_pairs, unresolved, alphabet, rules):
    while len(ext_pairs) > 0:
        pair = ext_pairs.pop()
        pair[0] = rules[pair[0]]
        pair[1] = rules[pair[1]]
        logger.log(check_specific_pair, f"Checking for exterior critical pairs between {pair[0]} and {pair[1]}")
        # We're looking for a proper suffix of l1 to be a prefix of l2
        for index in range(1, len(pair[0][1])):
            if pair[0][1][index:] == pair[1][1][:len(pair[0][1]) - index]:
                word1 = pair[0][1][:index] + pair[1][2]
                word2 = pair[0][2] + pair[1][1][len(pair[0][1]) - index:]
                # Prefix language is (L2 / pair[0][1][:index]) \cap L1
                prefixes = copy.deepcopy(pair[1][0])
                prefixes = quotient(prefixes, single_word_FSA(alphabet, pair[0][1][:index]))
                prefixes = intersection(prefixes, pair[0][0])
                prefixes = BFS(prefixes)
                if word1 != word2 and len(prefixes.accepts) > 0:
                    p = []
                    while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                        p.append(word1.pop(0))
                        del word2[0]
                    if len(p) > 0:
                        prefixes = concatenation(prefixes, single_word_FSA(prefixes.alphabet, p))
                    if len(prefixes.accepts) > 0 and word1 != word2:
                        logger.log(add_equation, f"Adding equation {word1} = {word2} after {prefixes}")
                        unresolved.append([prefixes, word1, word2])

def concat_prod_words_diag(word1, word2, alphabet):
    # This is necessary to prevent padding issues in a couple of places.
    # In particular, taking the product of word1 and word2 and then 
    # concatenating the diagonal on the left leads to padding symbols 
    # in the middle of the language if you use the basic method for 
    # concatenation.
    # That's *bad*, and ~~as far as I know there isn't an efficient way~~ 
    # it is impossible to translate from asynchronous to synchronous.
    # Hence this function, which builds the FSA for the synchronous version.
    alphabet_copy = list(copy.deepcopy(alphabet))
    alph = []
    for let1 in alphabet:
        for let2 in alphabet:
            alph.append((let1, let2))
        alph.append((let1, None))
        alph.append((None, let1))
    transitions = {}
    forced_both_state_count = min(len(word1), len(word2))
    forced_one_move_count = abs(len(word1) - len(word2))
    alph_size = len(alphabet)
    memory_state_count = alph_size * ((1 - alph_size ** forced_one_move_count) // (1 - alph_size))
    states = forced_both_state_count + 2 * memory_state_count + 3 - alph_size ** forced_one_move_count
    accepts = [states - 2]
    for let in alph:
        transitions[let] = [states - 1] * states
    for i in range(forced_both_state_count):
        transitions[(word1[i], word2[i])][i] = i + 1
    if len(word1) < len(word2):
        for i in range(len(word2) - len(word1)):
            forced_letter = word2[i + len(word1)]
            offset = forced_both_state_count + (1 - alph_size ** i) // (1 - alph_size)
            next_offset = offset + alph_size ** i
            for j in range(offset, next_offset):
                k = (j - offset) * alph_size + next_offset
                for location, letter in enumerate(alphabet_copy):
                    transitions[(letter, forced_letter)][j] = k + location
        for j in range(next_offset, next_offset + alph_size ** forced_one_move_count):
            k = (j - next_offset)
            oldest_memorized_index = k // (alph_size ** (forced_one_move_count - 1))
            oldest_memorized_letter = alphabet_copy[oldest_memorized_index]
            k -= oldest_memorized_index * (alph_size ** (forced_one_move_count - 1))
            k *= alph_size
            for location, letter in enumerate(alphabet_copy):
                transitions[(letter, oldest_memorized_letter)][j] = location + k + next_offset
        for i in range(len(word2) - len(word1)):
            offset = next_offset
            next_offset = offset + alph_size ** (forced_one_move_count - i)
            for j in range(offset, next_offset):
                k = j - offset
                oldest_memorized_index = k // (alph_size ** (forced_one_move_count - i - 1))
                oldest_memorized_letter = alphabet_copy[oldest_memorized_index]
                k -= oldest_memorized_index * (alph_size ** (forced_one_move_count - i - 1))
                transitions[(None, oldest_memorized_letter)][j] = next_offset + k
    elif len(word1) > len(word2):
        for i in range(len(word1) - len(word2)):
            forced_letter = word1[i + len(word2)]
            offset = forced_both_state_count + (1 - alph_size ** i) // (1 - alph_size)
            next_offset = offset + alph_size ** i
            for j in range(offset, next_offset):
                k = (j - offset) * alph_size + next_offset
                for location, letter in enumerate(alphabet_copy):
                    transitions[(forced_letter, letter)][j] = k + location
        for j in range(next_offset, next_offset + alph_size ** forced_one_move_count):
            k = (j - next_offset)
            oldest_memorized_index = k // (alph_size ** (forced_one_move_count - 1))
            oldest_memorized_letter = alphabet_copy[oldest_memorized_index]
            k -= oldest_memorized_index * (alph_size ** (forced_one_move_count - 1))
            k *= alph_size
            for location, letter in enumerate(alphabet_copy):
                transitions[(oldest_memorized_letter, letter)][j] = location + k + next_offset
        for i in range(len(word1) - len(word2)):
            offset = next_offset
            next_offset = offset + alph_size ** (forced_one_move_count - i)
            for j in range(offset, next_offset):
                k = j - offset
                oldest_memorized_index = k // (alph_size ** (forced_one_move_count - i - 1))
                oldest_memorized_letter = alphabet_copy[oldest_memorized_index]
                k -= oldest_memorized_index * (alph_size ** (forced_one_move_count - i - 1))
                transitions[(oldest_memorized_letter, None)][j] = next_offset + k
    else:
        for letter in alphabet_copy:
            transitions[(letter, letter)][forced_both_state_count] = forced_both_state_count
    return FSA.FSA(states, accepts, alph, transitions)

def check_pre_pairs(pre_pairs, unresolved, alphabet, everything, rules):
    while len(pre_pairs) > 0:
        pair = pre_pairs.pop()
        pair[0] = rules[pair[0]]
        pair[1] = rules[pair[1]]
        logger.log(check_specific_pair, f"Checking for prefix critical pairs between {pair[0]} and {pair[1]}")
        prefixes = copy.deepcopy(pair[0][0])
        prefixes = product(prefixes, prefixes)
        prefixes = intersection(prefixes, diagonal(alphabet))
        prefixes = concatenation(prefixes, concat_prod_words_diag(pair[0][1], pair[0][2], alphabet))
        # The previous line was a nightmare. There was a bug that lasted for 
        # about two years because of that step. Anyway, bug fixed!
        prefixes = intersection(prefixes, product(pair[1][0], complement(pair[1][0])))
        prefixes = projection(prefixes, [1])
        if len(prefixes.accepts) > 0:
            word1 = copy.copy(pair[1][1])
            word2 = copy.copy(pair[1][2])
            p = []
            while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                p.append(word1.pop(0))
                del word2[0]
            if len(p) > 0:
                prefixes = concatenation(prefixes, single_word_FSA(prefixes.alphabet, p))
            if len(prefixes.accepts) > 0 and word1 != word2:
                logger.log(add_equation, f"Adding equation {word1} = {word2} after {prefixes}")
                unresolved.append([prefixes, word1, word2])

def find_descendents(equation, rules):
    irreducibles = []
    checked = []
    toBeChecked = [[equation[0], equation[1]], [equation[0], equation[2]]]
    while len(toBeChecked) > 0:
        [lang, word] = toBeChecked.pop()
        logger.log(logging.DEBUG, f"descending {word} after {lang}")
        if len(lang.accepts) == 0:
            continue
        checked.append([lang, word])
        for rule in rules:
            for index in range(0, len(word) - len(rule[1]) + 1):
                if word[index: index + len(rule[1])] == rule[1]:
                    prefixes = intersection(concatenation(lang, single_word_FSA(lang.alphabet, word[:index])), rule[0])
                    prefixes = quotient(prefixes, single_word_FSA(lang.alphabet, word[:index]))
                    lang = intersection(lang, complement(prefixes))
                    newWord = word[:index] + rule[2] + word[index + len(rule[1]):]
                    for pair in toBeChecked:
                        if pair[1] == newWord:
                            prefixes = intersection(prefixes, pair[0])
                    for pair in checked:
                        if pair[1] == newWord:
                            prefixes = intersection(prefixes, pair[0])
                    if len(prefixes.accepts) > 0:
                        toBeChecked.append([prefixes, newWord])
        if len(lang.accepts) > 0:
            irreducibles.append([lang, word])
    return irreducibles

def resolve_equalities(unresolved, rules, alph, ordering, int_pairs, ext_pairs, pre_pairs):
    new_unresolved = []
    while len(unresolved) > 0:
        equation = unresolved.pop()
        logger.log(handle_specific_equation, f"Working on equation: {equation[1]} = {equation[2]} after {equation[0]}")
        irreducibles = find_descendents(equation, rules)
        logger.log(handle_specific_equation, f"Orienting {len(irreducibles)} irreducibles")
        while len(irreducibles) > 0:
            [lang, word] = irreducibles.pop()
            for index in range(0, len(irreducibles)):
                lang1 = copy.deepcopy(lang)
                word1 = copy.deepcopy(word)
                [lang2, word2] = copy.deepcopy(irreducibles[index])
                logger.log(orient_specific_pair, f"Orienting {word} after {lang} against {word2} after {lang2}")
                if word1 == word2:
                    continue
                inter = intersection(lang1, lang2)
                if len(inter.accepts) == 0:
                    continue
                p = []
                while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                    p.append(word1.pop(0))
                    del word2[0]
                if len(p) > 0:
                    inter = concatenation(inter, single_word_FSA(inter.alphabet, p))
                (left, right, incomp) = ordering(word1, word2, inter)
                if len(incomp.accepts) > 0 and word1 != word2:
                    logger.log(equation_did_not_resolve, f"Returning unresolved equation: {word1} = {word2} after {incomp}")
                    new_unresolved.append([incomp, word1, word2])
                for rule in rules:
                    if rule[1] == word1:
                        if rule[2] == word2:
                            right = intersection(right, complement(rule[0]))
                    if rule[1] == word2:
                        if rule[2] == word1:
                            left = intersection(left, complement(rule[0]))
                if len(right.accepts) > 0:
                    for ruleIndex in range(len(rules)):
                        if len(word1) > len(rules[ruleIndex][1]):
                            int_pairs.append([ruleIndex, len(rules)])
                        elif len(word1) < len(rules[ruleIndex][1]):
                            int_pairs.append([len(rules), ruleIndex])
                        ext_pairs.append([ruleIndex, len(rules)])
                        ext_pairs.append([len(rules), ruleIndex])
                        pre_pairs.append([ruleIndex, len(rules)])
                        pre_pairs.append([len(rules), ruleIndex])
                    ext_pairs.append([len(rules), len(rules)])
                    pre_pairs.append([len(rules), len(rules)])
                    logger.log(add_rule, f"Adding rule {word1} -> {word2} after {right}")
                    rules.append([right, word1, word2])
                if len(left.accepts) > 0:
                    for ruleIndex in range(len(rules)):
                        if len(word2) > len(rules[ruleIndex][1]):
                            int_pairs.append([ruleIndex, len(rules)])
                        elif len(word2) < len(rules[ruleIndex][1]):
                            int_pairs.append([len(rules), ruleIndex])
                        ext_pairs.append([ruleIndex, len(rules)])
                        ext_pairs.append([len(rules), ruleIndex])
                        pre_pairs.append([ruleIndex, len(rules)])
                        pre_pairs.append([len(rules), ruleIndex])
                    ext_pairs.append([len(rules), len(rules)])
                    pre_pairs.append([len(rules), len(rules)])
                    logger.log(add_rule, f"Adding rule {word2} -> {word1} after {left}")
                    rules.append([left, word2, word1])
    return new_unresolved

def clean_equations(rules, unresolved, alph):
    for equation in unresolved:
        logger.log(logging.DEBUG, f"handling equation {equation[1]} = {equation[2]} after {equation[0]}")
        for rule in rules:
            logger.log(logging.DEBUG, f"handling rule {rule[1]} -> {rule[2]} after {rule[0]}")
            index = 0
            while index < len(equation[1]) - len(rule[1]) + 1:
                if rule[1] == equation[1][index: index + len(rule[1])]:
                    p = copy.copy(equation[1][0:index])
                    redlang = intersection(equation[0], quotient(rule[0], single_word_FSA(alph, p)))
                    if len(redlang.accepts) > 0:
                        s = copy.copy(equation[1][index + len(rule[1]):])
                        equation[0] = intersection(equation[0], complement(redlang))
                        logger.log(logging.DEBUG, f"replacing lang with {equation[0]}")
                        if p + rule[2] + s != equation[2]:
                            unresolved.append([redlang, p + rule[2] + s, equation[2]])
                            logger.log(logging.DEBUG, f"adding new equation {redlang, p + rule[2] + s, equation[2]}")
                            index = 0
                    else:
                        index += 1
                else:
                    index += 1
            index = 0
            while index < len(equation[2]) - len(rule[1]) + 1:
                if rule[1] == equation[2][index: index + len(rule[1])]:
                    p = copy.copy(equation[2][0:index])
                    redlang = intersection(equation[0], quotient(rule[0], single_word_FSA(alph, p)))
                    if len(redlang.accepts) > 0:
                        s = copy.copy(equation[2][index + len(rule[1]):])
                        equation[0] = intersection(equation[0], complement(redlang))
                        logger.log(logging.DEBUG, f"replacing lang with {equation[0]}")
                        if equation[1] != p + rule[2] + s:
                            unresolved.append([redlang, equation[1], p + rule[2] + s])
                            logger.log(logging.DEBUG, f"adding new equation {redlang, equation[1], p + rule[2] + s}")
                            index = 0
                    else:
                        index += 1
                else:
                    index += 1
    for i in range(len(unresolved) - 1):
        for j in range(i + 1, len(unresolved)):
            if unresolved[i][1] == unresolved[j][1] and unresolved[i][2] == unresolved[j][2]:
                unresolved[i][0] = FSA.union(unresolved[i][0], unresolved[j][0])
                unresolved[j][0].accepts = set()
            elif unresolved[i][1] == unresolved[j][2] and unresolved[i][2] == unresolved[j][1]:
                unresolved[i][0] = FSA.union(unresolved[i][0], unresolved[j][0])
                unresolved[j][0].accepts = set()
    for i in range(len(unresolved) - 1, -1, -1):
        if len(unresolved[i][0].accepts) == 0 or unresolved[i][1] == unresolved[i][2]:
            logger.log(logging.DEBUG, f"pruning {unresolved[i]}")
            del unresolved[i]

def prune_prefixes(unresolved, rules, alph, everything):
    new_unresolved = []
    while len(unresolved) > 0:
        equation = unresolved.pop()
        logger.log(handle_specific_equation, f"Rewriting prefixes in equation {equation[1]} = {equation[2]} after {equation[0]}")
        for rule in rules:
            L1byAll = product(equation[0], everything)
            L2byL2 = product(rule[0], rule[0])
            L2byL2capDStar = intersection(L2byL2, diagonal(alph))
            L2byL2capDStarDot = concatenation(L2byL2capDStar, concatenation(product(single_word_FSA(alph, rule[1]), single_word_FSA(alph, rule[2])), diagonal(alph)))
            L3 = intersection(L1byAll, L2byL2capDStarDot)
            equation[0] = intersection(equation[0], complement(projection(L3, [0])))
            equation[0] = union(equation[0], projection(L3, [1]))
            if len(equation[0].accepts) == 0:
                break
        if len(equation[0].accepts) > 0 and equation[1] != equation[2]:
            logger.log(equation_did_not_resolve, f"Returning equation {equation[1]} = {equation[2]} after {equation[0]}")
            new_unresolved.append(equation)
    return new_unresolved

def reduce_prefixes(unresolved, rules, alph):
    # This rewrites at the border between prefixes and words, i.e. for L: u = v
    # if there's a rule L_2: l -> r where the end of l is the start of u, 
    # and the start of l can be the end of something in L, then we do that
    # rewriting. It's... somewhat dangerous. If you tell return the new
    # equation to this, it's very easy to loop infinitely here. 
    new_unresolved = []
    while len(unresolved) > 0:
        equality = unresolved.pop()
        logger.log(handle_specific_equation, f"Rewriting with rules that overlap prefix and word in equation {equality[1]} = {equality[2]} after {equality[0]}")
        for rule in rules:
            for index in range(0, len(rule[1])):
                if rule[1][index:] == equality[1][:len(rule) - index]: # overlap with left side of equality
                    # p = rule[1][:index], o = rule[1][index:], s = equality[1][:len(rule) - index]
                    prefixes = intersection(rule[0], quotient(equality[0], single_word_FSA(alph, rule[1][:index])))
                    if len(prefixes.accepts) > 0:
                        word1 = rule[2] + equality[1][len(rule) - index:]
                        word2 = rule[1][:index] + equality[2]
                        while len(word1) > 0 and len(word2) > 0 and word1[-1] == word2[-1]:
                            del word1[-1]
                            del word2[-1]
                        p = []
                        while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                            p.append(word1.pop(0))
                            del word2[0]
                        if len(p) > 0:
                            prefixes = concatenation(prefixes, single_word_FSA(prefixes.alphabet, p))
                        if len(prefixes.accepts) > 0 and word1 != word2:
                            logger.log(equation_did_not_resolve, f"Adding equality {word1} = {word2} after {prefixes}")
                            new_unresolved.append([prefixes, word1, word2])
                        equality[0] = intersection(equality[0], complement(prefixes))
                        if len(equality[0].accepts) == 0:
                            break
                if rule[1][index:] == equality[2][:len(rule) - index]: # overlap with right side of equality
                    prefixes = intersection(rule[0], quotient(equality[0], single_word_FSA(alph, rule[1][:index])))
                    if len(prefixes.accepts) > 0:
                        word1 = rule[2] + equality[2][len(rule) - index:]
                        word2 = rule[1][:index] + equality[1]
                        while len(word1) > 0 and len(word2) > 0 and word1[-1] == word2[-1]:
                            del word1[-1]
                            del word2[-1]
                        p = []
                        while len(word1) > 0 and len(word2) > 0 and word1[0] == word2[0]:
                            p.append(word1.pop(0))
                            del word2[0]
                        if len(p) > 0:
                            prefixes = concatenation(prefixes, single_word_FSA(prefixes.alphabet, p))
                        if len(prefixes.accepts) > 0 and word1 != word2:
                            logger.log(equation_did_not_resolve, f"Adding equality {word1} = {word2} after {prefixes}")
                            new_unresolved.append([prefixes, word1, word2])
                        equality[0] = intersection(equality[0], complement(prefixes))
                        if len(equality[0].accepts) == 0:
                            break
        if len(equality[0].accepts) > 0 and equality[1] != equality[2]:
            logger.log(equation_did_not_resolve, f"Returning equality {equality[1]} = {equality[2]} after {equality[0]}")
            new_unresolved.append(equality)
    return new_unresolved


def check_rule_lengths(max_rule_length, unresolved):
    # Checks for any rules that are too long
    for equality in unresolved:
        if len(equality[1]) > max_rule_length:
            return True
        if len(equality[2]) > max_rule_length:
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
        nf = intersection(complement(concatenation(concatenation(rule[0], single_word_FSA(group.generators, rule[1])), ev)), nf)
    return nf

def pKB(group, max_rule_number = 1000, max_rule_length = None, max_time = 600):
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
            unresolved.append([everything, rel[0], rel[1]]) # Rules are of the form [L, l, r], meaning that after any word in L, l goes to r
    # Initial step complete. Now we loop between the (critpair check/equality resolution) loop and the (check unresolved/prefix reduction) loop.
    while True: # This type of loop (where we continue forever and break at a certain condition) is apparently called a "loop and a half"; python, unfortunately, lacks a native "until" loop type.
        # Check for critical pairs
        # Interior as top priority, just because they are quick to check and eliminate things
        # Exterior as a second priority, i.e. todo if there are no pairs that we haven't check for interior pairs. Harder to check, but not as bad as:
        # Prefix as a bottom priority. THese are *slow* to check, since it's all about constructing FSAs.
        if len(int_pairs) > 0:
            logger.log(major_steps, f"Checking {len(int_pairs)} pairs of rules for interior critical pairs.")
            check_int_pairs(int_pairs, ext_pairs, pre_pairs, unresolved, group.generators, rules)
        elif len(ext_pairs) > 0:
            logger.log(major_steps, f"Checking {len(ext_pairs)} pairs of rules for exterior critical pairs.")
            check_ext_pairs(ext_pairs, unresolved, group.generators, rules)
        elif len(pre_pairs) > 0:
            logger.log(major_steps, f"Checking {len(pre_pairs)} pairs of rules for prefix critical pairs.")
            check_pre_pairs(pre_pairs, unresolved, group.generators, everything, rules)
        # Equality resolution
        logger.log(periodic_rule_display, f"Rules are {rules}")
        logger.log(major_steps, f"Resolving {len(unresolved)} equations.")
        unresolved = resolve_equalities(unresolved, rules, group.generators, group.ordering, int_pairs, ext_pairs, pre_pairs)
        # We've run the equality resolution; need to now check if unresolved is empty, and do prefix resolution step if not
        # The logic here could almost certainly be made more efficient. I expect that prefix resolution is slow, and doing it less often would likely be preferable.
        # But I'm forcing myself to remember that this is proof of concept, not finished product. Efficiencies can be made later.
        if len(unresolved) > 0:
            logger.log(major_steps, f"Pruning and reducing {len(unresolved)} unresolved equations.")
            logger.log(logging.DEBUG, "start clean_equations")
            clean_equations(rules, unresolved, group.generators)
            logger.log(logging.DEBUG, "start prune_prefixes")
            unresolved = prune_prefixes(unresolved, rules, group.generators, everything)
            # It turns out that reducing prefixes at the boundary of a word is a bad idea, actually.
            # At least from initial testing. This needs to be treated *carefully*.
            #logger.log(logging.DEBUG, "start reduce_prefixes")
            #unresolved = reduce_prefixes(unresolved, rules, group.generators)
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
