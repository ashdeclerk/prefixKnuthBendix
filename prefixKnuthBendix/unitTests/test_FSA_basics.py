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

class TestRepresentations:
    def test_repr_1(self):
        f = FSA(3, {0, 1}, ('a', 'b', 'c'), {'a': [0, 0, 1], 'b': [0, 0, 1], 'c': [0, 2, 0]})
        assert repr(f) == "FSA(3, {0, 1}, ('a', 'b', 'c'), {'a': [0, 0, 1], 'b': [0, 0, 1], 'c': [0, 2, 0]})"

    def test_repr_2(self):
        pass

    def test_repr_3(self):
        pass

    def test_string_1(self):
        pass

    def test_string_2(self):
        pass

    def test_string_3(self):
        pass

class TestEquality:
    def test_equality_1(self):
        pass
    
    def test_equality_2(self):
        pass
    
    def test_equality_3(self):
        pass

class TestWordArithmetic:
    def test_accept_reject_1(self):
        pass

    def test_accept_reject_2(self):
        pass

    def test_accept_reject_3(self):
        pass

    def test_target_1(self):
        pass

    def test_target_2(self):
        pass

    def test_target_3(self):
        pass

class TestManualManipulation:
    def test_add_letter(self):
        pass

    def test_remove_letter(self):
        pass

    def test_add_state(self):
        pass

    def test_remove_state(self):
        pass

    def test_change_edge(self):
        pass

    def test_change_init(self):
        pass
