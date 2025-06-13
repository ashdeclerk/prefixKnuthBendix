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
        self.accepts = list(accepts)
        self.alphabet = tuple(sorted(alphabet))
        self.transitions = copy.deepcopy(transitions)

    def __repr__(self):
        out = f"FSA({self.states}, {self.accepts}, {self.alphabet}, {self.transitions})"
        return out

    def __str__(self):
        out = f"Number of states: {self.states}\n"
        out += f"Accepting states: {self.accepts}\n"
        out += f"Alphabet: {self.alphabet}\n"
        out += "Transitions: {\n"
        for letter in self.transitions:
            out += f"    {letter}: {self.transitions[letter]}\n"
        out += "}"
        return out
    
    def __eq__(self, other):
        if isinstance(other, FSA):
            if other.states != self.states:
                return False
            if sorted(other.alphabet) != sorted(self.alphabet):
                return False
            if other.states != self.states:
                return False
            if other.transitions != self.transitions:
                return False
            return True
        return False

    def accepts_word(self, word):
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

    def target_state_from_location(self, word, location):
        for letter in word:
            location = self.transitions[letter][location]
        return location

    def add_letter(self, letter):
        if letter not in self.alphabet:
            self.alphabet = self.alphabet + (letter,)
            self.transitions[letter] = [0] * self.states

    def remove_letter(self, letter):
        if letter in self.alphabet:
            self.alphabet = tuple(l for l in self.alphabet if l != letter)
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

    def change_edge(self, start, end, label):
        self.transitions[label][start] = end

    def add_accepting_state(self, state):
        if state in self.accepts:
            return
        self.accepts.append(state)
    
    def remove_accepting_state(self, state):
        while state in self.accepts:
            self.accepts.remove(state)

    def change_init(self, index):
        if index in self.accepts:
            if 0 in self.accepts:
                pass
            else:
                self.remove_accepting_state(index)
                self.add_accepting_state(0)
        else:
            if 0 in self.accepts:
                self.remove_accepting_state(0)
                self.add_accepting_state(index)
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
    accepts: tuple
    alphabet: tuple
    transitions: frozenset


def freeze(fsa):
    transitions = {}
    for key, item in fsa.transitions.items():
        transitions[key] = tuple(item)
    transitions = frozenset(transitions.items())
    return frozenFSA(states = fsa.states, accepts = tuple(fsa.accepts), alphabet = fsa.alphabet, transitions = transitions)

    
def complement(fsa):
    acc = set(range(fsa.states))
    acc.difference_update(fsa.accepts)
    return BFS(FSA(fsa.states, acc, fsa.alphabet, fsa.transitions))


def product(fsa1, fsa2):
    states = (fsa1.states + 1) * (fsa2.states + 1)
    accepts = set()
    for i in fsa1.accepts:
        for j in fsa2.accepts:
            accepts.update({i * (fsa2.states + 1) + j})
        accepts.update({(i + 1) * (fsa2.states + 1) - 1})
    for j in fsa2.accepts:
        accepts.update({(fsa1.states) * (fsa2.states + 1) + j})
    arity1 = fsa1.alphabet[0]
    if type(arity1) == tuple:
        arity1 = len(arity1)
    else:
        arity1 = 1
    arity2 = fsa2.alphabet[0]
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
        list1.extend([''] * arity2)
        lettuple = tuple(list1)
        alphabet.update({lettuple})
        transitions[lettuple] = [states - 1] * states
        for i in range(0, fsa1.states):
            for j in fsa2.accepts:
                transitions[lettuple][i * (fsa2.states + 1) + j] = (fsa1.transitions[let1][i] + 1) * (fsa2.states + 1) - 1
            transitions[lettuple][(i + 1) * (fsa2.states + 1) - 1] = (fsa1.transitions[let1][i] + 1) * (fsa2.states + 1) - 1
    for let2 in fsa2.alphabet:
        list1 = [''] * arity1
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
    return BFS(FSA(states, accepts, alphabet, transitions))

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
            if '' in let:
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
                epsilonTargets.update(self.transitions[''][state])
        for letter in word:
            newLocations = set()
            newLocationsPreUpdate = set()
            for state in locations:
                newLocations.update(self.transitions[letter][state])
            while newLocations != newLocationsPreUpdate:
                newLocationsPreUpdate = newLocations
                for state in newLocationsPreUpdate:
                    newLocations.update(self.transitions[''][state])
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
        self.transitions[''].append(set())

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
            del self.transitions[''][index]
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
    accepts = set(fsa.accepts)
    alphabet = fsa.alphabet
    transitions = {}
    transitions[''] = [set()] * states
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
        if letter == '':
            return True
        if type(letter) != tuple:
            return False
        for entry in letter:
            if entry != '':
                return False
        return True
    
    alphabet = nfa.alphabet
    accept_states = set()
    transitions = {}
    nonish_letters = []
    alphabet = tuple(let for let in alphabet if not is_nonish(let))
    for let in alphabet:
        transitions[let] = []
    starting_states = {0}
    potential_starting_states = {0}
    while len(potential_starting_states) > 0:
        # We need to find epsilon transitions from our current list of potential starting states.
        new_potential_starting_states = set()
        for letter in nonish_letters:
            for state in potential_starting_states:
                new_potential_starting_states.update(nfa.transitions[letter][state])
        new_potential_starting_states.difference_update(starting_states)
        starting_states.update(new_potential_starting_states)
        potential_starting_states = new_potential_starting_states
    # Now we need to actually build the rest of the fsa.
    # Each state in the fsa corresponds to a set of states in the nfa.
    # The dictionary `state_map` is just telling us that correspondence. 
    state_map = {0: starting_states} 
    # `current_state` tells us which state of the fsa we're working with
    current_state = 0
    # and `state_count` tells us how many states of the fsa we know about
    state_count = 1
    while current_state < state_count:
        # Step 1: Figure out the states of the nfa each letter sends current_state to.
        for letter in alphabet:
            target_states = set()
            for nfa_state in state_map[current_state]:
                target_states.update(nfa.transitions[letter][nfa_state])
            # Step 1.1: Deal with epsilon transitions
            potential_target_states = copy.copy(target_states)
            while len(potential_target_states) > 0:
                new_potential_target_states = set()
                for epsilon in nonish_letters:
                    for nfa_state in potential_target_states:
                        new_potential_target_states.update(nfa.transitions[epsilon][nfa_state])
                new_potential_target_states.difference_update(target_states)
                target_states.update(new_potential_target_states)
                potential_target_states = copy.copy(new_potential_target_states)
            # Step 1.2: Check if this is an old state and handle transitions appropriately
            is_old = False
            for fsa_state in range(state_count):
                if state_map[fsa_state] == target_states:
                    transitions[letter].append(fsa_state)
                    is_old = True
            if not is_old:
                transitions[letter].append(state_count)
                state_map[state_count] = target_states
                state_count += 1
        # Then we move on to the next state!
        current_state += 1
    # We have to deal with accept states!
    for state in range(state_count):
        if state_map[state].intersection(nfa.accepts):
            accept_states.update({state})
    out = BFS(FSA(state_count, accept_states, alphabet, transitions))
    return out

def star(fsa):
    out = copy.deepcopy(fsa)
    out.add_accepting_state(0)
    out = nondeterminize(out)
    for letter in out.alphabet:
        for state in fsa.accepts:
            out.transitions[letter][state].update(out.transitions[letter][0])
    out = determinize(out)
    return out

def union(f: FSA, g: FSA):
    # Rewriting this to avoid NFAs.
    if sorted(f.alphabet) != sorted(g.alphabet):
        raise AlphabetError(f.alphabet, g.alphabet, f"In union, {f} and {g} have different alphabets.")
    processed_states = 0
    states = 1
    state_number_to_state_pairs = [(0, 0)]
    state_pairs_to_state_number = {(0, 0): 0}
    transitions = {}
    alph = sorted(f.alphabet)
    accepting_states = []
    if 0 in f.accepts or 0 in g.accepts:
        accepting_states.append(0)
    for let in alph:
        transitions[let] = []
    while processed_states < states:
        state_to_process = state_number_to_state_pairs[processed_states]
        for let in alph:
            next_state = (f.transitions[let][state_to_process[0]], g.transitions[let][state_to_process[1]])
            if next_state in state_pairs_to_state_number:
                next_state_number = state_pairs_to_state_number[next_state]
            else:
                state_number_to_state_pairs.append(next_state)
                state_pairs_to_state_number[next_state] = states
                next_state_number = states
                if next_state[0] in f.accepts or next_state[1] in g.accepts:
                    accepting_states.append(next_state_number)
                states += 1
            transitions[let].append(next_state_number)
        processed_states += 1
    return BFS(FSA(states, accepting_states, alph, transitions))

def intersection(f, g):
    # Rewriting this to avoid NFAs.
    if sorted(f.alphabet) != sorted(g.alphabet):
        raise AlphabetError(f.alphabet, g.alphabet, f"In intersection, {f} and {g} have different alphabets.")
    processed_states = 0
    states = 1
    state_number_to_state_pairs = [(0, 0)]
    state_pairs_to_state_number = {(0, 0): 0}
    transitions = {}
    alph = sorted(f.alphabet)
    accepting_states = []
    if 0 in f.accepts and 0 in g.accepts:
        accepting_states.append(0)
    for let in alph:
        transitions[let] = []
    while processed_states < states:
        state_to_process = state_number_to_state_pairs[processed_states]
        for let in alph:
            next_state = (f.transitions[let][state_to_process[0]], g.transitions[let][state_to_process[1]])
            if next_state in state_pairs_to_state_number:
                next_state_number = state_pairs_to_state_number[next_state]
            else:
                state_number_to_state_pairs.append(next_state)
                state_pairs_to_state_number[next_state] = states
                next_state_number = states
                if next_state[0] in f.accepts and next_state[1] in g.accepts:
                    accepting_states.append(next_state_number)
                states += 1
            transitions[let].append(next_state_number)
        processed_states += 1
    return BFS(FSA(states, accepting_states, alph, transitions))

def quotient(fsa1, fsa2):
    # Note that there are two types of quotients here. This computes 
    # {x | there exists y in L2 such that xy is in L1}.
    # For {x | xy is in L1 for all y in L2} you want strict_quotient.
    if sorted(fsa1.alphabet) != sorted(fsa2.alphabet):
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
    return BFS(FSA(outStates, outAccepts, outAlphabet, outTransitions))

def strict_quotient(fsa1, fsa2):
    if sorted(fsa1.alphabet) != sorted(fsa2.alphabet):
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
    return BFS(FSA(outStates, outAccepts, outAlphabet, outTransitions))

def singleton_quotient(fsa, word):
    # I realized that doing a full quotient is inefficient.
    out_states = fsa.states
    out_accepts = []
    out_alphabet = fsa.alphabet
    out_transitions = fsa.transitions
    for state in range(0, fsa.states):
        if fsa.target_state_from_location(word, state) in fsa.accepts:
            out_accepts.append(state)
    return BFS(FSA(out_states, out_accepts, out_alphabet, out_transitions))

def concatenation(f, g):
    if sorted(f.alphabet) != sorted(g.alphabet):
        raise AlphabetError(f.alphabet, g.alphabet, f"In concatenation, {f} and {g} have different alphabets.")
    processed_states = 0
    states = 1
    accepting_states = []
    if 0 in f.accepts:
        starting_pair = (0, (0,))
        if 0 in g.accepts:
            accepting_states.append(0)
    else:
        starting_pair = (0, ())
    state_number_to_state_pairs = [starting_pair]
    state_pairs_to_state_number = {starting_pair: 0}
    transitions = {}
    alph = sorted(f.alphabet)
    for let in alph:
        transitions[let] = []
    while processed_states < states:
        state_to_process = state_number_to_state_pairs[processed_states]
        for let in alph:
            next_f_state = f.transitions[let][state_to_process[0]]
            next_g_states = list(set((g.transitions[let][g_state] for g_state in state_to_process[1])))
            if next_f_state in f.accepts and 0 not in next_g_states:
                next_g_states.append(0)
            next_state = (next_f_state, tuple(sorted(next_g_states)))
            if next_state in state_pairs_to_state_number:
                transitions[let].append(state_pairs_to_state_number[next_state])
            else:
                state_number_to_state_pairs.append(next_state)
                state_pairs_to_state_number[next_state] = states
                transitions[let].append(states)
                if any((s in g.accepts for s in next_state[1])):
                    accepting_states.append(states)
                states += 1
        processed_states += 1
    return BFS(FSA(states, accepting_states, alph, transitions))

def diagonal(alph):
    states = 2
    accepts = {0}
    alphabet = set()
    for let1 in alph:
        for let2 in alph:
            alphabet.update({(let1, let2)})
        alphabet.update({(let1, '')})
        alphabet.update({('', let1)})
    transitions = {}
    for let in alphabet:
        transitions[let] = [1, 1]
    for let in alph:
        transitions[(let, let)][0] = 0
    return FSA(states, accepts, alphabet, transitions)

def projection(fsa, indices):
    alph = []
    letter_map = {}
    if len(indices) > 1:
        null_letter = ('',) * len(indices)
        letter_map[null_letter] = []
        for let in fsa.alphabet:
            new_let = tuple(let[i] for i in indices)
            if new_let not in alph and new_let != null_letter:
                alph.append(new_let)
                letter_map[new_let] = []
            letter_map[new_let].append(let)
    else:
        null_letter = ''
        letter_map[null_letter] = []
        for let in fsa.alphabet:
            new_let = let[indices[0]]
            if new_let not in alph and new_let != null_letter:
                alph.append(new_let)
                letter_map[new_let] = []
            letter_map[new_let].append(let)
    alph = sorted(alph)
    # We've handled the alphabet. Now we need to handle the transitions, states, and accepting states.
    states = 1
    processed_states = 0
    accepting_states = []
    # Dealing with epsilon transitions is a huge pain in the butt.
    starting_state_description = [0]
    i = 0
    while i < len(starting_state_description):
        state_to_check = starting_state_description[i]
        for let in letter_map[null_letter]:
            target = fsa.transitions[let][state_to_check]
            if target not in starting_state_description:
                starting_state_description.append(target)
        i += 1
    starting_state_description = tuple(sorted(starting_state_description))
    if any((s in fsa.accepts for s in starting_state_description)):
        accepting_states.append(0)
    state_number_to_state_description = [starting_state_description]
    state_description_to_state_number = {starting_state_description: 0}
    transitions = {}
    for let in alph:
        transitions[let] = []
    while processed_states < states:
        current_state_description = state_number_to_state_description[processed_states]
        for let in alph:
            target_state_description = []
            for s in current_state_description:
                for super_letter in letter_map[let]:
                    next_state = fsa.transitions[super_letter][s]
                    if next_state not in target_state_description:
                        target_state_description.append(next_state)
            # I hate epsilon transitions *so much*.
            i = 0
            while i < len(target_state_description):
                state_to_check = target_state_description[i]
                for epsilon in letter_map[null_letter]:
                    target = fsa.transitions[epsilon][state_to_check]
                    if target not in target_state_description:
                        target_state_description.append(target)
                i += 1
            target_state_description = tuple(sorted(target_state_description))
            if target_state_description in state_description_to_state_number:
                transitions[let].append(state_description_to_state_number[target_state_description])
            else:
                if any((s in fsa.accepts for s in target_state_description)):
                    accepting_states.append(states)
                transitions[let].append(states)
                state_description_to_state_number[target_state_description] = states
                state_number_to_state_description.append(target_state_description)
                states += 1
        processed_states += 1
    return BFS(FSA(states, accepting_states, alph, transitions))

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
    transitions[''] = {}
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
            transitions[''][grammar.variables.index(rule[0])].update({grammar.variables.index(rule[1][-1])})
    return NFA(states, accepts, grammar.terminals, transitions)

def reverse(nfa):
    states = nfa.states + 1
    transitions = {}
    for let in nfa.alphabet:
        transitions[let] = {}
    transitions[''] = {}
    for let in transitions.keys():
        for state in range(states):
            transitions[let][state] = set()
        for state in range(nfa.states):
            for state2 in nfa.transitions[let][state]:
                transitions[let][state2 + 1].update({state + 1})
    for state in nfa.accepts:
        transitions[''][0].update({state + 1})
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
        pre.accepts = {state}
        pre = FSA.remove_padded_words(pre)
        post0 = copy.deepcopy(fsa)
        post0.change_init(state)
        post0.add_state()
        for let in fsa.alphabet:
            post0.changeEdge(post0.states, post0.states, let)
        post1 = copy.deepcopy(post0)
        for let in fsa.alphabet:
            if let[1] != '':
                for state2 in range(post0.states):
                    post0.changeEdge(state2, post0.states, let)
            if let[0] != '':
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

def singletons_diagonal_concatenate(word1, word2, alph):
    # This returns the FSA accepting (word1 x word2) . D*
    ordered_alph = list(alph)
    if len(ordered_alph) == 1:
        a = ordered_alph[0]
        squared_alph = [(a, a), (a, ''), ('', a)]
        minlen = min(len(word1), len(word2))
        maxlen = max(len(word1), len(word2))
        states = maxlen + 1
        accepts = {maxlen}
        transitions = {}
        for let in squared_alph:
            transitions[let] = [states - 1] * states
        for i in range(min(len(word1), len(word2))):
            transitions[(a, a)][i] = i + 1
        transitions[(a, a)][min(len(word1), len(word2))] = min(len(word1), len(word2))
        if len(word1) > len(word2):
            for i in range(maxlen - minlen):
                transitions[(a, '')][minlen + i] = minlen + i + 1
        elif len(word2) > len(word1):
            for i in range(maxlen - minlen):
                transitions[('', a)][minlen + i] = minlen + i + 1
        return FSA(states, accepts, squared_alph, transitions)
    squared_alph = []
    transitions = {}
    minlen = min(len(word1), len(word2))
    diff = max(len(word1), len(word2)) - minlen
    # There are minlen states leading up to lattice of memory states
    # There are 1 + |alph| + ... + |alph|^diff + ... + |alph| + 1 such memory states
    # And there's one fail state.
    states = minlen + (len(alph) ** (diff + 1) - 1) // (len(alph) - 1) + 1
    accepts = {states - 2}
    for let1 in alph:
        for let2 in alph:
            transitions[(let1, let2)] = [states - 1] * states
            squared_alph.append((let1, let2))
        transitions[(let1, '')] = [states - 1] * states
        squared_alph.append((let1, ''))
        transitions[('', let1)] = [states - 1] * states
        squared_alph.append(('', let1))
    for index in range(minlen - 1):
        transitions[(word1[index], word2[index])][index] = index + 1
    # At this point we are in the memory lattice.
    # Conveniently, the smallest state of the lattice is state minlen.
    # So, the memory lattice is essentially keeping track of "here's what the 
    # longer word is owed". We're encoding that as a base len(alph) number
    # and adding that to the starting state of the layer.
    # Note that this encoding is most-urgently-needed letter in the units place,
    # so if ordered_alph is [a, b] and the word we're encoding is abb, that
    # gets encoded in binary as 110 (i.e. the 'a' in the units place).
    # We also have to send state minlen - 1 to the appropriate state in
    # the first layer of the memory lattice.
    layer_start = minlen
    next_layer_start = minlen + len(alph) ** diff
    starting_debt = 0
    if len(word1) > len(word2):
        for index, letter in enumerate(word1[minlen:]):
            starting_debt += ordered_alph.index(letter) * len(alph) ** index
    elif len(word1) < len(word2):
        for index, letter in enumerate(word2[minlen:]):
            starting_debt += ordered_alph.index(letter) * len(alph) ** index
    if minlen > 0:
        transitions[(word1[minlen - 1], word2[minlen - 1])][minlen - 1] = minlen + starting_debt
    if len(word1) > len(word2):
        # The first layer has an extra thing to worry about compared to all
        # of the other layers. Namely, we have to worry about wandering between
        # the first layer rather than moving on to the second layer.
        for state_index in range(next_layer_start - layer_start):
            current_state = state_index + layer_start
            oldest_letter = ordered_alph[state_index % len(alph)]
            base_state = layer_start + state_index // len(alph)
            for new_letter_index, new_letter in enumerate(ordered_alph):
                transitions[(oldest_letter, new_letter)][current_state] = base_state + new_letter_index * len(alph) ** (diff - 1)
        for layer_index in range(diff):
            # Now we have to deal with going from one layer to the next.
            for state_index in range(next_layer_start - layer_start):
                current_state = state_index + layer_start
                oldest_letter = ordered_alph[state_index % len(alph)]
                transitions[(oldest_letter, '')][current_state] = next_layer_start + state_index // len(alph)
            layer_start = next_layer_start
            next_layer_start = next_layer_start + len(alph) ** (diff - layer_index - 1)
    if len(word2) > len(word1):
        # Similar, but with words swapped
        for state_index in range(next_layer_start - layer_start):
            current_state = state_index + layer_start
            oldest_letter = ordered_alph[state_index % len(alph)]
            base_state = layer_start + state_index // len(alph)
            for new_letter_index, new_letter in enumerate(ordered_alph):
                transitions[(new_letter, oldest_letter)][current_state] = base_state + new_letter_index * len(alph) ** (diff - 1)
        for layer_index in range(diff):
            for state_index in range(next_layer_start - layer_start):
                current_state = state_index + layer_start
                oldest_letter = ordered_alph[state_index % len(alph)]
                transitions[('', oldest_letter)][current_state] = next_layer_start + state_index // len(alph)
            layer_start = next_layer_start
            next_layer_start = next_layer_start + len(alph) ** (diff - layer_index - 1)
    if len(word1) == len(word2):
        # No memory needed, we just loop on (l, l).
        for let in alph:
            transitions[(let, let)][minlen] = minlen
    out = FSA(states, accepts, squared_alph, transitions)
    if minlen == 0:
        # Oops, special case.
        out.change_init(starting_debt)
    return out
