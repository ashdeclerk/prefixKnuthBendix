# PrefixKB

PrefixKB implements prefix-Knuth-Bendix (a procedure to search for 
autostackable structures) in Python. For a description of prefix-Knuth-Bendix,
please see Ash DeClerk's PhD dissertation.

To install this program, make sure you've installed Python 3 (so far this has 
been tested on Python 3.10) and download the repository. You should either add 
the prefixKnuthBendix folder to your Python path or keep working files in its 
parent directory (i.e. the way the repo is currently set up -- with 
`Cox333pKB.py` in the folder that contains the prefixKnuthBendix folder).

I highly recommend reading through the examples `Cox333pKB.py` and `BS12pKB.py`
(ideally in that order) to see how to set up a group for pKB. You can also email
me at ash.declerk@gmail.com if you need help.

# TODO
- [ ] Set up proper testing for the various submodules
- [ ] Make a standalone "tell me what group and ordering you want"
  - [ ] From file as command line arg
  - [ ] From command line prompting (as gens/rels/ord or as file)
- [ ] Make a log parser
- [ ] Add "pick back up from where this log left off" functionality
- [ ] Write the paper (not really a programming thing, but still important)
- [ ] Add functionality to AutostackableStructure
  - [ ] Normal forms
  - [ ] Rewriting words
- [ ] THEN you get to play with knot groups and whatever else, Ash.
