This branch is primarily to make the code more object-oriented. In particular:
- [ ] Make a class `Equation`
- [ ] Make a class `Rule`
- [ ] Make a class `RewritingChain`
- [ ] Implement these classes appropriately
- [ ] Change all references to rules/equations to use these classes

There are two major benefits here: First (and the reason I'm doing this now), it
will hopefully simplify log parsing because I'll be tracking "oh, this rule came
from these parents"; Second, it will allow heuristics-based custom implementation
of pKB. (The general implementation here is a bit naive, and may be slower than
other strategies for how to order steps for certain groups -- or even not work
when a different strategy does, if things go very badly.)