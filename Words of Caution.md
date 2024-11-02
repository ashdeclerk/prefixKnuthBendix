# Words of Caution

For anyone using the FSA class here for their own purposes, there are a few bits
to be careful about, especially if you're coming from other FSA implementations.

- The starting state of an FSA is state 0.
- FSAs are currently always fully described, i.e. there's no default target 
  for a transition. 
- If there is a fail state, it's explicitly listed as a state rather than 
  being something like state -1.
- The transition function is given as a nested dictionary. The first layer of
  keys is the letter, the second layer of keys is the current state, and the
  entry is the new state. That is, transitions[letter][location] is the state
  after reading letter at location.
- Letters can be any immutable. Strings are fine. Numbers are fine. You could,
  in theory, use a frozen FSA as a letter. (This is inadvisable, but if you
  have a specific use in mind, go for it.)
- `None` is reserved as a padding symbol! Please don't use it as a letter.
  (And also don't use tuples as letters unless your alphabet is genuinely
  a product alphabet. That could screw some things up.)
- Adding a state to an FSA that already exists (via the `addState` method)
  defaults to every letter transitioning from the new state to the starting
  state. Make sure to change edges appropriately after adding a state.
- Synchronous regular languages are not closed under concatenation in general,
  but they do play nice with specifically concatenation by a single word. 
  The function `sync_singleton_concatenate` does this concatenation. (It didn't
  do the thing I actually needed when I wrote it, but it still works, so I'm
  leaving it in.)
