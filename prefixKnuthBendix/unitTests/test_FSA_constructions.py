from ..FSA.FSA import *

"""
This file tests FSA constructions, i.e.
- Unions
- Concatenations (note that this won't play nice with synchronous multi-tape FSAs)
- Intersections
- Kleene stars
- Products
- Projections
- FSAs accepting a single given word
- Complements
- Breadth-first-search normal forms for FSAs
- Quotients
- Strict quotients (in the sense of )
- Substitutions
- Inverse homomorphisms
- Reversing FSAs (note that this won't play nice with synchronous multi-tape FSAs)
- Concatenation of a synchronous multi-tape FSA with a single word (this *does* work)
- FSA accepting the synchronous language {(uw, vw) | w in A*} for specific u, v
"""

# Randomly generated FSAs to used for testing
f = FSA(3, {0, 1}, ('a', 'b', 'c'), {'a': [0, 0, 1], 'b': [0, 0, 1], 'c': [0, 2, 0]})
g = FSA(7, {2, 3, 6}, ('a', 'b', 'c'), {'a': [3, 5, 3, 5, 6, 4, 3], 'b': [6, 2, 1, 1, 6, 5, 1], 'c': [5, 0, 2, 2, 3, 4, 4]})
h = FSA(11, {0, 1, 5, 7, 9}, ('a', 'b', 'c'), {'a': [5, 6, 0, 4, 8, 6, 2, 8, 8, 4, 3], 'b': [1, 1, 4, 4, 1, 8, 10, 3, 5, 5, 4], 'c': [8, 5, 7, 7, 6, 7, 1, 7, 2, 2, 8]})

# Non-random FSA to check a specific thing in BFS
i = FSA(5, {3, 4}, ('a', 'b', 'c'), {'a': [1, 2, 2, 1, 2], 'b': [2, 4, 3, 2, 2], 'c': [3, 3, 4, 3, 3]})

# Randomly generated short words to be used for testing
u = ['a', 'a', 'b']
v = ['a', 'b', 'a', 'c', 'c', 'a']
w = ['b', 'a', 'b']

class TestBFS:
    def test_BFS_1(self):
        assert BFS(f) == FSA(1, {0}, ('a', 'b', 'c'), {'a': [0], 'b': [0], 'c': [0]})
    
    def test_BFS_2(self):
        assert BFS(g) == FSA(7, {1, 2, 5}, ('a', 'b', 'c'), {'a': [1, 3, 1, 6, 3, 1, 2], 'b': [2, 4, 4, 3, 5, 4, 2], 'c': [3, 5, 6, 6, 0, 5, 1]})

    def test_BFS_3(self):
        assert BFS(h) == FSA(10, {0, 1, 2, 5}, ('a', 'b', 'c'), {'a': [1, 4, 4, 3, 6, 3, 0, 8, 9, 3], 'b': [2, 3, 2, 1, 7, 8, 9, 9, 9, 2], 'c': [3, 5, 1, 6, 2, 5, 5, 3, 5, 4]})

    def test_BFS_4(self):
        assert BFS(i) == FSA(3, {2}, ('a', 'b', 'c'), {'a': [1, 1, 1], 'b': [1, 2, 1], 'c': [2, 2, 2]})

class TestUnion:
    def test_union_1(self):
        assert union(f, g) == FSA(1, {0}, ('a', 'b', 'c'), {'a': [0], 'b': [0], 'c': [0]})

    def test_union_2(self):
        assert union (g, g) == BFS(g)

    def test_union_3(self):
        # This took me an hour and a half to calculate by hand.
        assert union (g, h) == BFS(FSA(56, {0, 1, 2, 6, 7, 8, 9, 11, 14, 15, 17, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 34, 35, 36 ,37, 38, 39, 40, 43, 45, 46, 47, 48, 49, 50, 51, 54, 55}, ('a', 'b', 'c'), 
                                       {'a': [1, 4, 7, 10, 12,  3, 17, 19,  4, 23, 24, 28, 30, 32, 23,  7, 34,  3, 33, 40, 41,  7,  7, 27, 17,  3,  7, 43, 45, 24,  1, 17, 31, 10, 11, 10,  4, 34, 17, 17, 26, 52, 17,  9,  3, 34, 54,  4, 55, 28, 17, 27, 24, 27, 33,  3],
                                        'b': [2, 5, 8, 11, 13, 15, 18, 20, 21, 24, 26,  3, 31, 33,  2,  5, 31, 36, 38, 33, 38,  8, 24, 20, 36, 18,  5, 44, 46, 48,  8,  8, 31, 49,  8, 41, 50, 44,  8, 48,  2, 33, 26, 49, 21, 44, 44,  8, 44, 49, 36, 20,  2, 46, 44,  8],
                                        'c': [3, 6, 9, 12, 14, 16,  6, 21, 22, 25, 27, 29, 25, 10,  1,  6, 35, 37, 39, 29, 42, 15, 35, 14, 12,  6, 29,  6, 47, 25, 10, 28, 25, 28, 50, 29, 39,  6, 51, 35, 17, 29, 19, 10, 53, 29, 10, 15, 29,  9, 37, 21,  7, 49,  6, 51]}))

class TestIntersection:
    def test_intersection_1(self):
        assert intersection(f, g) == BFS(g)
    
    def test_intersection_2(self):
        assert intersection(g, g) == BFS(g)

    def test_intersection_3(self):
        # Yay for using essentially the same work twice!
        assert union (g, h) == BFS(FSA(56, {34, 30, 21, 47, 2, 15, 1, 26, 6, 25}, ('a', 'b', 'c'), 
                                       {'a': [1, 4, 7, 10, 12,  3, 17, 19,  4, 23, 24, 28, 30, 32, 23,  7, 34,  3, 33, 40, 41,  7,  7, 27, 17,  3,  7, 43, 45, 24,  1, 17, 31, 10, 11, 10,  4, 34, 17, 17, 26, 52, 17,  9,  3, 34, 54,  4, 55, 28, 17, 27, 24, 27, 33,  3],
                                        'b': [2, 5, 8, 11, 13, 15, 18, 20, 21, 24, 26,  3, 31, 33,  2,  5, 31, 36, 38, 33, 38,  8, 24, 20, 36, 18,  5, 44, 46, 48,  8,  8, 31, 49,  8, 41, 50, 44,  8, 48,  2, 33, 26, 49, 21, 44, 44,  8, 44, 49, 36, 20,  2, 46, 44,  8],
                                        'c': [3, 6, 9, 12, 14, 16,  6, 21, 22, 25, 27, 29, 25, 10,  1,  6, 35, 37, 39, 29, 42, 15, 35, 14, 12,  6, 29,  6, 47, 25, 10, 28, 25, 28, 50, 29, 39,  6, 51, 35, 17, 29, 19, 10, 53, 29, 10, 15, 29,  9, 37, 21,  7, 49,  6, 51]}))