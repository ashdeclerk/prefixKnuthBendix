from ..FSA.FSA import *

"""
This file tests the "basics" of finite state automata, i.e.
- creation and representation
- string representation
- accepting words
- finding target states
- adding/removing letters
- adding/removing states
- changing edges
- changing initial vertices
- checking equality of two (potentially different-looking) FSAs
"""


# Randomly generated FSAs to used for testing
f = FSA(3, {0, 1}, ('a', 'b', 'c'), {'a': [0, 0, 1], 'b': [0, 0, 1], 'c': [0, 2, 0]})
g = FSA(7, {2, 3, 6}, ('a', 'b', 'c'), {'a': [3, 5, 3, 5, 6, 4, 3], 'b': [6, 2, 1, 1, 6, 5, 1], 'c': [5, 0, 2, 2, 3, 4, 4]})
h = FSA(11, {0, 1, 5, 7, 9}, ('a', 'b', 'c'), {'a': [5, 6, 0, 4, 8, 6, 2, 8, 8, 4, 3], 'b': [1, 1, 4, 4, 1, 8, 10, 3, 5, 5, 4], 'c': [8, 5, 7, 7, 6, 7, 1, 7, 2, 2, 8]})

# Randomly generated words to used for testing
u = ['a', 'b', 'a', 'c', 'c', 'c', 'c', 'b', 'c', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b']
v = ['b', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'a', 'b', 'a', 'c', 'a', 'a', 'c', 'b', 'b']
w = ['a', 'a', 'a', 'a', 'c', 'a', 'c', 'a', 'b', 'c', 'a', 'a', 'c', 'c', 'c', 'a']

class TestRepresentations:
    def test_repr_1(self):
        assert repr(f) == "FSA(3, [0, 1], ('a', 'b', 'c'), {'a': [0, 0, 1], 'b': [0, 0, 1], 'c': [0, 2, 0]})"

    def test_repr_2(self):
        assert repr(g) == "FSA(7, [2, 3, 6], ('a', 'b', 'c'), {'a': [3, 5, 3, 5, 6, 4, 3], 'b': [6, 2, 1, 1, 6, 5, 1], 'c': [5, 0, 2, 2, 3, 4, 4]})"

    def test_repr_3(self):
        assert repr(h) == "FSA(11, [0, 1, 5, 7, 9], ('a', 'b', 'c'), {'a': [5, 6, 0, 4, 8, 6, 2, 8, 8, 4, 3], 'b': [1, 1, 4, 4, 1, 8, 10, 3, 5, 5, 4], 'c': [8, 5, 7, 7, 6, 7, 1, 7, 2, 2, 8]})"

    def test_string_1(self):
        assert str(f) == """Number of states: 3
Accepting states: [0, 1]
Alphabet: ('a', 'b', 'c')
Transitions: {
    a: [0, 0, 1]
    b: [0, 0, 1]
    c: [0, 2, 0]
}"""

    def test_string_2(self):
        assert str(g) == """Number of states: 7
Accepting states: [2, 3, 6]
Alphabet: ('a', 'b', 'c')
Transitions: {
    a: [3, 5, 3, 5, 6, 4, 3]
    b: [6, 2, 1, 1, 6, 5, 1]
    c: [5, 0, 2, 2, 3, 4, 4]
}"""

    def test_string_3(self):
        assert str(h) == """Number of states: 11
Accepting states: [0, 1, 5, 7, 9]
Alphabet: ('a', 'b', 'c')
Transitions: {
    a: [5, 6, 0, 4, 8, 6, 2, 8, 8, 4, 3]
    b: [1, 1, 4, 4, 1, 8, 10, 3, 5, 5, 4]
    c: [8, 5, 7, 7, 6, 7, 1, 7, 2, 2, 8]
}"""

class TestEquality:
    def test_equality_1(self):
        assert f != g
    
    def test_equality_2(self):
        assert h == h
    
    def test_equality_3(self):
        i = FSA(3, {1, 0}, ('b', 'a', 'c'), {'c': [0, 2, 0], 'a': [0, 0, 1], 'b': [0, 0, 1]})
        assert f == i

class TestWordArithmetic:
    def test_accept_reject_1(self):
        assert f.accepts_word(u)

    def test_accept_reject_2(self):
        assert g.accepts_word(v)

    def test_accept_reject_3(self):
        assert not h.accepts_word(w)

    def test_target_1(self):
        assert f.target_state(v) == 0

    def test_target_2(self):
        assert g.target_state(w) == 3

    def test_target_3(self):
        assert h.target_state(u) == 8

class TestManualManipulation:
    def test_add_letter(self):
        i = copy.deepcopy(f)
        i.add_letter('d')
        assert i == FSA(3, {0, 1}, ('a', 'b', 'c', 'd'), {'a': [0, 0, 1], 'b': [0, 0, 1], 'c': [0, 2, 0], 'd': [0, 0, 0]})

    def test_remove_letter(self):
        i = copy.deepcopy(f)
        i.remove_letter('a')
        assert i == FSA(3, {0, 1}, ('b', 'c'), {'b': [0, 0, 1], 'c': [0, 2, 0]})

    def test_add_state(self):
        i = copy.deepcopy(f)
        i.add_state()
        assert i == FSA(4, {0, 1}, ('a', 'b', 'c'), {'a': [0, 0, 1, 0], 'b': [0, 0, 1, 0], 'c': [0, 2, 0, 0]})

    def test_remove_state(self):
        i = copy.deepcopy(f)
        i.remove_state(1)
        assert i == FSA(2, {0}, ('a', 'b', 'c'), {'a': [0, 1], 'b': [0, 1], 'c': [0, 0]})

    def test_change_edge(self):
        i = copy.deepcopy(f)
        i.change_edge(0, 2, 'a')
        assert i == FSA(3, {0, 1}, ('a', 'b', 'c'), {'a': [2, 0, 1], 'b': [0, 0, 1], 'c': [0, 2, 0]})

    def test_change_init(self):
        i = copy.deepcopy(f)
        i.change_init(2)
        assert i == FSA(3, {1, 2}, ('a', 'b', 'c'), {'a': [1, 2, 2], 'b': [1, 2, 2], 'c': [2, 0, 2]})
