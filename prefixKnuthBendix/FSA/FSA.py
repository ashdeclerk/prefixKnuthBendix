import copy
import random
from dataclasses import dataclass

class AlphabetError(Exception):
    """
    Exception raised when alphabets don't match.
    """

    def __init__(self, alph1, alph2, message):
        self.alph1 = alph1
        self.alph2 = alph2
        self.message = message



class FSA:

    def __init__(self, states, accepts, alphabet, transitions):
        self.states = states
        self.accepts = copy.deepcopy(accepts)
        self.alphabet = set(copy.deepcopy(alphabet))
        self.transitions = copy.deepcopy(transitions)

    def __repr__(self):
        out = f"Number of states: {self.states}\n"
        out += f"Accepting states: {self.accepts}\n"
        out += f"Alphabet: {self.alphabet}\n"
        out += "Transitions: \n"
        for letter in self.transitions:
            out += f"\t {letter}: {self.transitions[letter]}\n"
        return out

    def accepted(self, word):
        location = 0
        for letter in word:
            location = self.transitions[letter][location]
        if location in self.accepts:
            return True
        return False
    
    def target_state(self, word):
        location = 0
        for letter in word:
            location = self.transitions[letter][location]
        return location

    def add_letter(self, letter):
        if letter not in self.alphabet:
            self.alphabet.update({letter})
            self.transitions[letter] = [0] * self.states

    def remove_letter(self, letter):
        if letter in self.alphabet:
            self.alphabet.difference_update({letter})
            del self.transitions[letter]

    def add_state(self):
        self.states += 1
        for letter in self.alphabet:
            self.transitions[letter].append(0)

    def remove_state(self, index):
        if index <= self.states:
            self.states -= 1
            for letter in self.alphabet:
                del self.transitions[letter][index]
                for i in range(0, self.states):
                    if self.transitions[letter][i] > index:
                        self.transitions[letter][i] -= 1

    def changeEdge(self, start, end, label):
        self.transitions[label][start] = end

    def change_init(self, index):
        if index in self.accepts:
            if 0 in self.accepts:
                pass
            else:
                self.accepts.difference_update({index})
                self.accepts.update({0})
        else:
            if 0 in self.accepts:
                self.accepts.difference_update({0})
                self.accepts.update({index})
        for letter in self.alphabet:
            self.transitions[letter][0], self.transitions[letter][index] = self.transitions[letter][index], self.transitions[letter][0]
            for state in range(0, self.states):
                if self.transitions[letter][state] == 0:
                    self.transitions[letter][state] = index
                elif self.transitions[letter][state] == index:
                    self.transitions[letter][state] = 0


@dataclass(frozen = True)
class frozenFSA:

    states: int
    accepts: frozenset
    alphabet: frozenset
    transitions: frozenset


def freeze(fsa):
    transitions = {}
    for key, item in fsa.transitions.items():
        transitions[key] = tuple(item)
    transitions = frozenset(transitions.items())
    return frozenFSA(states = fsa.states, accepts = frozenset(fsa.accepts), alphabet = frozenset(fsa.alphabet), transitions = transitions)

    
def complement(fsa):
    acc = set(range(fsa.states))
    acc.difference_update(fsa.accepts)
    return FSA(fsa.states, acc, fsa.alphabet, fsa.transitions)


def product(fsa1, fsa2):
    states = (fsa1.states + 1) * (fsa2.states + 1)
    accepts = set()
    for i in fsa1.accepts:
        for j in fsa2.accepts:
            accepts.update({i * (fsa2.states + 1) + j})
        accepts.update({(i + 1) * (fsa2.states + 1) - 1})
    for j in fsa2.accepts:
        accepts.update({(fsa1.states) * (fsa2.states + 1) + j})
    arity1 = copy.deepcopy(fsa1.alphabet)
    arity1 = arity1.pop()
    if type(arity1) == tuple:
        arity1 = len(arity1)
    else:
        arity1 = 1
    arity2 = copy.deepcopy(fsa2.alphabet)
    arity2 = arity2.pop()
    if type(arity2) == tuple:
        arity2 = len(arity2)
    else:
        arity2 = 1
    alphabet = set()
    transitions = {}
    for let1 in fsa1.alphabet:
        for let2 in fsa2.alphabet:
            if type(let1) == tuple:
                list1 = list(let1)
            else:
                list1 = [let1]
            if type(let2) == tuple:
                list1.extend(let2)
            else:
                list1 += [let2]
            lettuple = tuple(list1)
            alphabet.update({lettuple})
            transitions[lettuple] = [states - 1] * states
            for i in range(0, fsa1.states):
                for j in range(0, fsa2.states):
                    transitions[lettuple][i * (fsa2.states + 1) + j] = fsa1.transitions[let1][i] * (fsa2.states + 1) + fsa2.transitions[let2][j]
        if type(let1) == tuple:
            list1 = list(let1)
        else:
            list1 = [let1]
        list1.extend([None] * arity2)
        lettuple = tuple(list1)
        alphabet.update({lettuple})
        transitions[lettuple] = [states - 1] * states
        for i in range(0, fsa1.states):
            for j in fsa2.accepts:
                transitions[lettuple][i * (fsa2.states + 1) + j] = (fsa1.transitions[let1][i] + 1) * (fsa2.states + 1) - 1
            transitions[lettuple][(i + 1) * (fsa2.states + 1) - 1] = (fsa1.transitions[let1][i] + 1) * (fsa2.states + 1) - 1
    for let2 in fsa2.alphabet:
        list1 = [None] * arity1
        if type(let2) == tuple:
            list1.extend(let2)
        else:
            list1.append(let2)
        lettuple = tuple(list1)
        alphabet.update({lettuple})
        transitions[lettuple] = [states - 1] * states
        for i in fsa1.accepts:
            for j in range(0, fsa2.states):
                transitions[lettuple][i * (fsa2.states + 1) + j] = (fsa1.states) * (fsa2.states + 1) + fsa2.transitions[let2][j]
        for j in range(0, fsa2.states):
            transitions[lettuple][(fsa1.states) * (fsa2.states + 1) + j] = (fsa1.states) * (fsa2.states + 1) + fsa2.transitions[let2][j]
    return FSA(states, accepts, alphabet, transitions)

def clean_padding(FSA):
    arity = copy.deepcopy(FSA.alphabet)
    arity = arity.pop()
    if type(arity) != tuple:
        return
    pass

def remove_padded_words(fsa):
    out = copy.deepcopy(fsa)
    out.add_state()
    for let in out.alphabet:
        out.changeEdge(out.states, out.states, let)
        if type(let) == tuple:
            if None in let:
                for state in range(out.states):
                    out.changeEdge(state, out.states, let)
    return out

def single_word_FSA(alph, word):
    for let in word:
        if let not in alph:
            raise AlphabetError(alph, alph, "In single_word_FSA, " + str(word) + " was not a word over " + str(alph))
    states = len(word) + 2
    alphabet = alph
    transitions = {}
    for let in alphabet:
        transitions[let] = [states - 1] * (states)
    for index in range(0, len(word)):
        transitions[word[index]][index] = index + 1
    accepts = {states - 2}
    return FSA(states, accepts, alphabet, transitions)

def empty_FSA(alph):
    states = 1
    alphabet = alph
    transitions = {}
    for let in alphabet:
        transitions[let] = [0]
    accepts = set()
    return FSA(states, accepts, alphabet, transitions)

def all_FSA(alph):
    states = 1
    alphabet = alph
    transitions = {}
    for let in alphabet:
        transitions[let] = [0]
    accepts = {0}
    return FSA(states, accepts, alphabet, transitions)

def intersect_lists(list1, list2):
    out = []
    for entry in list1:
        if entry in list2:
            out.append(entry)
    return out

def subtract_list(list1, list2):
    out = []
    for entry in list1:
        if entry not in list2:
            out.append(entry)
    return out

def BFS(fsa):
    alphabet = fsa.alphabet
    P = []
    W = []
    if len(fsa.accepts) > 0:
        P.append(fsa.accepts)
        W.append(fsa.accepts)
    nonAcc = set(range(0, fsa.states)).difference(fsa.accepts)
    if len(nonAcc) > 0:
        P.append(nonAcc)
        W.append(nonAcc)
    del nonAcc
    while len(W) > 0:
        A = W.pop(0)
        for let in alphabet:
            X = []
            for state in range(0, fsa.states):
                if fsa.transitions[let][state] in A:
                    X.append(state)
            for Y in P:
                if len(intersect_lists(Y, X)) > 0 and len(subtract_list(Y, X)) > 0:
                    P.remove(Y)
                    P.append(intersect_lists(Y, X))
                    P.append(subtract_list(Y, X))
                    if W.count(Y) > 0:
                        W.remove(Y)
                        W.append(intersect_lists(Y, X))
                        W.append(subtract_list(Y, X))
                    else:
                        W.append(intersect_lists(Y, X))
    states = len(P)
    accepts = set()
    transitions = {}
    for let in alphabet:
        transitions[let] = [0] * (states)
    for i in range(0, len(P)):
        if 0 in P[i]:
            P.insert(0, P.pop(i))
    for i in range(0, len(P)):
        if list(P[i])[0] in fsa.accepts:
            accepts.update({i})
        for let in alphabet:
            for j in range(0, len(P)):
                if fsa.transitions[let][list(P[i])[0]] in P[j]:
                    transitions[let][i] = j
    relabeling = {0: 0}
    unrelabeling = {0: 0}
    location = 0
    counter = 1
    newStates = states
    while counter <= len(P):
        for let in alphabet:
            if transitions[let][location] not in relabeling:
                relabeling[transitions[let][location]] = counter
                unrelabeling[counter] = transitions[let][location]
                counter += 1
        location = relabeling[location] + 1
        if location >= counter:
            newStates = counter
            break
        location = unrelabeling[location]
    newAccepts = set()
    for state in accepts:
        if state in relabeling.keys():
            newAccepts.update({relabeling[state]})
    newTransitions = {}
    for let in alphabet:
        newTransitions[let] = [0] * (newStates)
        for state in range(0, states):
            if state in relabeling.keys():
                newTransitions[let][relabeling[state]] = relabeling[transitions[let][state]]
    return FSA(newStates, newAccepts, alphabet, newTransitions)



class NFA:

    def __init__(self, states, accepts, alphabet, transitions):
        self.states = states
        self.accepts = copy.deepcopy(accepts)
        self.alphabet = copy.deepcopy(alphabet)
        self.transitions = copy.deepcopy(transitions)

    def __repr__(self):
        out = "Number of states: " + str(self.states) + "\n"
        out += "Accepting states: " + str(self.accepts) + "\n"
        out += "Alphabet: " + str(self.alphabet) + "\n"
        out += "Transitions: \n"
        for letter in self.transitions:
            out += "\t " + str(letter) + ": " + str(self.transitions[letter]) + "\n"
        return out

    def accepted(self, word):
        locations = set()
        epsilonTargets = {0}
        while epsilonTargets != locations:
            locations = epsilonTargets
            for state in locations:
                epsilonTargets.update(self.transitions[None][state])
        for letter in word:
            newLocations = set()
            newLocationsPreUpdate = set()
            for state in locations:
                newLocations.update(self.transitions[letter][state])
            while newLocations != newLocationsPreUpdate:
                newLocationsPreUpdate = newLocations
                for state in newLocationsPreUpdate:
                    newLocations.update(self.transitions[None][state])
            locations = newLocations
        for state in locations:
            if self.accepts.count(state) > 0:
                return True
        return False

    def add_letter(self, letter):
        if letter not in self.alphabet:
            self.alphabet.update({letter})
            self.transitions[letter] = []
            for state in range(self.states):
                self.transitions[letter].append(set())

    def remove_letter(self, letter):
        if letter in self.alphabet:
            self.alphabet.difference_update({letter})
            del self.transitions[letter]

    def add_state(self):
        self.states += 1
        for letter in self.alphabet:
            self.transitions[letter].append(set())
        self.transitions[None].append(set())

    def remove_state(self, index):
        if index <= self.states:
            self.states -= 1
            newAccepts = set()
            for state in self.accepts:
                if state < index:
                    newAccepts.update({state})
                if state > index:
                    newAccepts.update({state - 1})
            self.accepts = newAccepts
            for letter in self.alphabet:
                del self.transitions[letter][index]
                for i in range(0, self.states):
                    oldStates = self.transitions[letter][index].intersection(set(range(index + 1, self.states + 1)))
                    newStates = set()
                    for location in oldStates:
                        newStates.update(location - 1)
                    self.transitions[letter][index].intersection_update(set(range(0, index)))
                    self.transitions[letter][index].union_update(newStates)
            del self.transitions[None][index]
            for i in range(0, self.states):
                oldStates = self.transitions[letter][index].intersection(set(range(index + 1, self.states + 1)))
                newStates = set()
                for location in oldStates:
                    newStates.update(location - 1)
                self.transitions[letter][index].intersection_update(set(range(0, index)))
                self.transitions[letter][index].union_update(newStates)


    def add_edge(self, start, end, label):
        self.transitions[label][start].update({end})

    def remove_edge(self, start, end, label):
        self.transitions[label][state].difference_update({end})

def nondeterminize(fsa):
    states = fsa.states
    accepts = fsa.accepts
    alphabet = fsa.alphabet
    transitions = {}
    transitions[None] = [set()] * states
    for letter in alphabet:
        transitions[letter] = [set()] * states
        for state in range(0, states):
            transitions[letter][state] = set([fsa.transitions[letter][state]])
    return NFA(states, accepts, alphabet, transitions)

def determinize(nfa):
    def is_nonish(letter):
        # This is a stupid hack to get around tracking the arity of FSAs.
        # If I were to rewrite this module from scratch, I would track
        # the arity of automata. And separate synchronous from asynchronous
        # for arity > 1, but that's a separate issue.
        if letter == None:
            return True
        if type(letter) != tuple:
            return False
        for entry in letter:
            if entry != None:
                return False
        return True
    
    alphabet = nfa.alphabet
    accepts = set()
    transitions = {}
    for let in alphabet:
        transitions[let] = [0]
    currentState = 0
    stateCount = 1
    start = {0}
    potential_starts = {0}
    while potential_starts != set():
        new_potential_starts = set()
        for let in alphabet:
            if is_nonish(let):
                for state in potential_starts:
                    new_potential_starts.update(nfa.transitions[let][state])
        new_potential_starts.difference_update(start)
        start.update(new_potential_starts)
        potential_starts = new_potential_starts
    stateLabels = {0: tuple(start)}
    labelStates = {tuple(start): 0}
    for i in start:
        if i in nfa.accepts:
            accepts.update({0})
    while True:
        for let in alphabet:
            targets = set()
            accepting = False
            for source in stateLabels[currentState]:
                targets.update(nfa.transitions[let][source])
            if targets.intersection(set(nfa.accepts)) != set():
                accepting = True
            potential_targets = targets
            while potential_targets != set():
                new_potential_targets = set()
                if let in alphabet:
                    if is_nonish(let):
                        for state in potential_targets:
                            new_potential_targets.update(nfa.transitions[let][state])
                new_potential_targets.difference_update(targets)
                targets.update(new_potential_targets)
                potential_targets = new_potential_targets
            targets = tuple(targets)
            if targets not in labelStates:
                labelStates[targets] = stateCount
                stateLabels[stateCount] = targets
                transitions[let][currentState] = stateCount
                if accepting:
                    accepts.update({stateCount})
                stateCount += 1
                for let in alphabet:
                    transitions[let].append(0)
            else:
                transitions[let][currentState] = labelStates[targets]
        currentState += 1
        if currentState >= stateCount:
            break
    nonish_letters = set()
    for let in alphabet:
        if is_nonish(let):
            nonish_letters.update({let})
    alphabet.difference_update(nonish_letters)
    out = FSA(stateCount, accepts, alphabet, transitions)
    return out

def star(fsa):
    out = nondeterminize(fsa)
    out.accepts.update({0})
    for letter in out.alphabet:
        for state in fsa.accepts:
            out.transitions[letter][state].update(out.transitions[letter][0])
    out = determinize(out)
    return out

def union(fsa1, fsa2):
    if fsa1.states == 1:
        if len(fsa1.accepts) == 1:
            return fsa1
        if len(fsa1.accepts) == 0:
            return fsa2
    if fsa2.states == 1:
        if len(fsa2.accepts) == 1:
            return fsa2
        if len(fsa2.accepts) == 0:
            return fsa1
    if fsa1.alphabet != fsa2.alphabet:
        raise AlphabetError(fsa1.alphabet, fsa2.alphabet, "In union, " + str(fsa1) + " and " + str(fsa2) + " have different alphabets.")
    nfa1 = nondeterminize(fsa1)
    nfa2 = nondeterminize(fsa2)
    for state in range(0, nfa2.states + 1):
        nfa1.add_state()
    if 0 in fsa1.accepts:
        nfa1.accepts.update({fsa1.states + fsa2.states})
    elif 0 in fsa2.accepts:
        nfa1.accepts.update({0})
    for state in nfa2.accepts:
        nfa1.accepts.update({state + fsa1.states})
    for letter in nfa1.alphabet:
        for target in nfa2.transitions[letter][0]:
            nfa1.transitions[letter][0].update([target + fsa1.states])
        for initial in range(0, nfa2.states):
            for target in nfa2.transitions[letter][initial]:
                nfa1.transitions[letter][initial + fsa1.states].update([target + fsa1.states])
        target = fsa1.transitions[letter][0]
        if target != 0:
            nfa1.transitions[letter][nfa1.states - 1].update([fsa1.transitions[letter][0]])
        else:
            nfa1.transitions[letter][nfa1.states - 1].update([nfa1.states - 1])
        for source in range(0, fsa1.states):
            if 0 in nfa1.transitions[letter][source]:
                nfa1.transitions[letter][source].symmetric_difference_update({0, nfa1.states - 1})
    out = BFS(determinize(nfa1))
    return out

def intersection(fsa1, fsa2):
    if fsa1.states == 1:
        if len(fsa1.accepts) == 1:
            return fsa2
        if len(fsa1.accepts) == 0:
            return fsa1
    if fsa2.states == 1:
        if len(fsa2.accepts) == 1:
            return fsa1
        if len(fsa2.accepts) == 0:
            return fsa2
    if fsa1.alphabet != fsa2.alphabet:
        raise AlphabetError(fsa1.alphabet, fsa2.alphabet, "In intersection, " + str(fsa1) + " and " + str(fsa2) + " have different alphabets.")
    states = fsa1.states * fsa2.states
    accepts = set()
    alphabet = fsa1.alphabet
    transitions = {}
    for i in fsa1.accepts:
        for j in fsa2.accepts:
            accepts.update({i * fsa2.states + j})
    for letter in alphabet:
        transitions[letter] = [0] * states
        for i in range(0, fsa1.states):
            for j in range(0, fsa2.states):
                transitions[letter][i * fsa2.states + j] = fsa1.transitions[letter][i] * fsa2.states + fsa2.transitions[letter][j]
    out = BFS(FSA(states, accepts, alphabet, transitions))
    return out

def quotient(fsa1, fsa2):
    if fsa1.alphabet != fsa2.alphabet:
        raise AlphabetError(fsa1.alphabet, fsa2.alphabet, "In quotient, " + str(fsa1) + " and " + str(fsa2) + " have different alphabets.")
    outStates = fsa1.states
    outAccepts = set()
    outAlphabet = fsa1.alphabet
    outTransitions = fsa1.transitions
    for state in range(0, fsa1.states):
        temp1 = copy.deepcopy(fsa1)
        temp1.change_init(state)
        temp1 = intersection(temp1, fsa2)
        temp1 = BFS(temp1)
        if len(temp1.accepts) > 0:
            outAccepts.update({state})
    return FSA(outStates, outAccepts, outAlphabet, outTransitions)

def strict_quotient(fsa1, fsa2):
    if fsa1.alphabet != fsa2.alphabet:
        raise AlphabetError(fsa1.alphabet, fsa2.alphabet, "In strict_quotient, " + str(fsa1) + " and " + str(fsa2) + " have different alphabets.")
    outStates = fsa1.states
    outAccepts = set()
    outAlphabet = fsa1.alphabet
    outTransitions = fsa1.transitions
    for state in range(0, fsa1.states):
        temp1 = copy.deepcopy(fsa1)
        temp1.change_init(state)
        temp1 = intersection(temp1, fsa2)
        temp1 = union(temp1, complement(fsa2))
        if temp1.states == 1:
            if temp1.accepts == [0]:
                outAccepts.update({state})
    return FSA(outStates, outAccepts, outAlphabet, outTransitions)

def concatenation(fsa1, fsa2):
    if fsa2.states == 1:
        if len(fsa2.accepts) == 0:
            return fsa2
    if fsa1.alphabet != fsa2.alphabet:
        raise AlphabetError(fsa1.alphabet, fsa2.alphabet, "In concatenation, " + str(fsa1) + " and " + str(fsa2) + " have different alphabets.")
    nfa1 = nondeterminize(fsa1)
    for i in range(fsa2.states):
        nfa1.add_state()
    for state in nfa1.accepts:
        for let in fsa1.alphabet:
            nfa1.transitions[let][state].update({fsa2.transitions[let][0] + fsa1.states})
    for state in fsa2.accepts:
        nfa1.accepts.update({state + fsa1.states})
    if 0 not in fsa2.accepts:
        for state in fsa1.accepts:
            nfa1.accepts.difference_update({state})
    for state in range(fsa2.states):
        for let in fsa1.alphabet:
            nfa1.transitions[let][state + fsa1.states].update([fsa2.transitions[let][state] + fsa1.states])
    out = BFS(determinize(nfa1))
    return out

def diagonal(alph):
    states = 2
    accepts = {0}
    alphabet = set()
    for let1 in alph:
        for let2 in alph:
            alphabet.update({(let1, let2)})
        alphabet.update({(let1, None)})
        alphabet.update({(None, let1)})
    transitions = {}
    for let in alphabet:
        transitions[let] = [1, 1]
    for let in alph:
        transitions[(let, let)][0] = 0
    return FSA(states, accepts, alphabet, transitions)
    
def projection(fsa, indices):
    out = nondeterminize(fsa)
    alphabet = set()
    transitions = {}
    for let in out.alphabet:
        if len(indices) > 1:
            newLet = tuple(let[i] for i in indices)
        else:
            newLet = let[indices[0]]
        if not newLet in alphabet:
            alphabet.update({newLet})
            transitions[newLet] = []
            for i in range(0, out.states):
                transitions[newLet].append(set())
        for state in range(0, out.states):
            transitions[newLet][state].update(out.transitions[let][state])
    out.alphabet = alphabet
    out.transitions = transitions
    return BFS(determinize(out))


def random_FSA(minStates, maxStates, acceptRate, alph):
    states = random.randint(minStates, maxStates)
    accepts = set()
    for i in range(0, states):
        if random.random() < acceptRate:
            accepts.update({i})
    transitions = {}
    for letter in alph:
        transitions[letter] = [0] * states
        for i in range(0, states):
            transitions[letter][i] = random.randint(0, states - 1)
    return FSA(states, accepts, alph, transitions)

def single_substitution(fsa1, letter, fsa2):
    nfa1 = nondeterminize(fsa1)
    for let in fsa2.alphabet:
        if let not in fsa1.alphabet:
            nfa1.add_letter(let)
    for source in range(fsa1.states):
        target = fsa1.transitions[letter][source]
        for state in range(fsa2.states):
            nfa1.add_state()
        for let in fsa2.alphabet:
            for s2 in range(fsa2.states):
                nfa1.transitions[let][nfa1.states - s2].update({nfa1.states - fsa2.transitions[let][s2]})
        for let in fsa2.alphabet:
            nfa1.transitions[let][source].update({nfa1.states - fsa2.transitions[let][0]})
        for s2 in fsa2.accepts:
            for let in fsa1.alphabet:
                nfa1.transitions[let][nfa1.states - s2].update({fsa1.transitions[let][target]})
    return BFS(determinize(nfa1))

def inverse_homomorphism(fsa, hom):
    alph = set(hom.keys())
    states = fsa.states
    accepts = fsa.accepts
    transitions = {}
    for let in alph:
        transitions[let] = [0] * states
        for source in range(states):
            target = source
            for let2 in hom[let]:
                target = fsa.transitions[let2][target]
            transitions[let][source] = target
    return BFS(FSA(states, accepts, alph, transitions))

class RegularGrammar:
    def __init__(self, variables, terminals, rules):
        self.variables = variables
        self.terminals = terminals
        self.rules = rules

def NFA_from_grammar(grammar):
    states = len(grammar.variables)
    accepts = set()
    transitions = {}
    for let in grammar.terminals:
        transitions[let] = {}
    transitions[None] = {}
    for let in transitions.keys():
        for state in range(states):
            transitions[let][state] = set()
    for rule in grammar.rules:
        currentState = grammar.variables.index(rule[0])
        if len(rule[1]) == 0:
            accepts.update([grammar.variables.index(rule[0])])
        elif len(rule[1]) > 2 and rule[1][-1] in grammar.variables:
            currentState = grammar.variables.index(rule[0])
            for i in range(len(rule[1]) - 2):
                states += 1
                transitions[rule[1][i]][currentState].update({states - 1})
                for let in transitions.keys():
                    transitions[let][states - 1] = set()
                currentState = states - 1
            transitions[rule[1][-2]][currentState].update({grammar.variables.index(rule[1][-1])})
        elif rule[1][-1] in grammar.terminals:
            currentState = grammar.variables.index(rule[0])
            for i in range(len(rule[1])):
                states += 1
                transitions[rule[1][i]][currentState].update({states - 1})
                for let in transitions.keys():
                    transitions[let][states - 1] = set()
                currentState = states - 1
            accepts.update([states - 1])
        elif len(rule[1]) == 2:
            transitions[rule[1][0]][grammar.variables.index(rule[0])].update({grammar.variables.index(rule[1][-1])})
        elif len(rule[1]) == 1:
            transitions[None][grammar.variables.index(rule[0])].update({grammar.variables.index(rule[1][-1])})
    return NFA(states, accepts, grammar.terminals, transitions)

def reverse(nfa):
    states = nfa.states + 1
    transitions = {}
    for let in nfa.alphabet:
        transitions[let] = {}
    transitions[None] = {}
    for let in transitions.keys():
        for state in range(states):
            transitions[let][state] = set()
        for state in range(nfa.states):
            for state2 in nfa.transitions[let][state]:
                transitions[let][state2 + 1].update({state + 1})
    for state in nfa.accepts:
        transitions[None][0].update({state + 1})
    return NFA(states, [1], nfa.alphabet, transitions)


class regularGrammar:

    def __init__(self, terminals, nonTerminals, rules, start):
        self.terminals = terminals
        self.nonTerminals = nonTerminals
        self.rules = rules

def right_grammar_to_FSA(grammar):
    alph = grammar.terminals
    states = {grammar.start: 0}
    i = 1
    for var in grammar.nonTerminals:
        if var != grammar.start:
            states[var] = i
            i += 1
    transitions = {}
    accepts = set()
    for letter in alph:
        transitions[letter] = [set()] * i
    for rule in grammar.rules:
        source = states[rule[0]]
        if rule[1] == []:
            accepts.update({source})
        else:
            pass
    pass

def sync_singleton_concatenate(fsa, word):
    transitions = {}
    for let in fsa.alphabet:
        transitions[let] = [0]
    out = FSA.FSA(1, {0}, fsa.alphabet, transitions)
    for state in range(fsa.states):
        pre = copy.deepcopy(fsa)
        pre.accepts = [state]
        pre = FSA.remove_padded_words(pre)
        post0 = copy.deepcopy(fsa)
        post0.change_init(state)
        post0.add_state()
        for let in fsa.alphabet:
            post0.changeEdge(post0.states, post0.states, let)
        post1 = copy.deepcopy(post0)
        for let in fsa.alphabet:
            if let[1] != None:
                for state2 in range(post0.states):
                    post0.changeEdge(state2, post0.states, let)
            if let[0] != None:
                for state2 in range(post1.states):
                    post1.changeEdge(state2, post1.states, let)
        post0 = projection(post0, 0)
        post1 = projection(post1, 1)
        post0 = concatenation(post0, single_word_FSA(word))
        post1 = concatenation(post0, single_word_FSA(word))
        post0 = product(post0, single_word_FSA(word))
        post1 = product(post1, single_word_FSA(word))
        out = union(out, concatenation(pre, union(post0, post1)))
    return out
