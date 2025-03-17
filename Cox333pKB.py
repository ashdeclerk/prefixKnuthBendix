# This is an example file for prefixKnuthBendix.
# This file finds an autostackable structure for the 3, 3, 3 Coxeter group.

# Lots of imports. I recommend not touching this.
from prefixKnuthBendix.prefixKnuthBendix import Group, pKB
from prefixKnuthBendix.FSA import FSA
from prefixKnuthBendix.orderAutomata import orderAutomata
from prefixKnuthBendix.pkbLogging import pkbLogging
import atexit
import logging.config
import logging.handlers
from queue import SimpleQueue

# This is a basic logging setup that logs absolutely everything.
logger = logging.getLogger("prefixKnuthBendix")
# If you want to log less things, you can increase the logging level.
# 11 logs almost everything, 21 or higher logs nothing,
# and various levels in between reduce the amount being logged.
# I recommend keeping this at 11 and parsing the log file later if necessary.
# 9 logs too much information. In particular, it logs every time an equation
# gets reduced, with one rule application per log. I don't recommend it unless
# you're planning to go really deep into the weeds of what's happening. I only
# added it because I needed to use that information to debug. 
logger.setLevel(11)
# You can comment out any keys that you don't want logged to the log file.
# Again, I recommend not touching this, because more information is generally
# better. But the message is probably enough information by itself.
format_keys = {
    "level": "levelname",
    "message": "message",
    "timestamp": "timestamp",
    "logger": "name",
    "module": "module",
    "function": "funcName",
    "line": "lineno",
    "thread_name": "threadName"
}
# You should change the file name in `make_file_handler` to an appropriate
# name for your log. I'm using a .jsonl (json lines) file because
# each line is formatted as a json object. File extensions are just a name, though. 
file_handler = pkbLogging.make_file_handler("Cox333.jsonl", level = 11, format_keys = format_keys)
# You can also adjust the logging levels separately for the file handler
# and stdout handler. If you want to just see the major steps in stdout but
# want everything logged to a file, you can set the stdout handler level to 
# 19, for example.
stdout_handler = pkbLogging.make_stdout_handler(level = 19)
# Both the file handler and stdout handler get thrown into a queue handler,
# because that has some effect on performance. Logging takes time, and 
# this puts logging on a separate thread.
# It's not a noticable amount of time for this small example,
# but more complex examples will likely benefit more.
log_queue = SimpleQueue()
queue_handler = logging.handlers.QueueHandler(log_queue)
queue_listener = logging.handlers.QueueListener(log_queue, file_handler, stdout_handler, respect_handler_level = True)
queue_listener.start()
atexit.register(queue_listener.stop)
logger.addHandler(queue_handler)

# And now we get to the actual pKB stuff!
# First we define the group we're interested in.
Cox333 = Group({'a', 'b', 'c'}, [[['a','a'],[]], [['b','b'],[]], [['c','c'],[]], [['a','b', 'a'], ['b', 'a', 'b']], [['a','c','a'],['c', 'a', 'c']], [['b', 'c', 'b'], ['c', 'b', 'c']]])
# Then we make the ordering. In this case we're using regular-split shortlex.
# If you want to use an ordering based on an FSA that isn't included in
# orderAutomata, you should have that FSA accept pairs (u, v) such that u > v.
# You don't need all such pairs to be accepted (in theory if pKB terminates
# with that ordering, it only needs a bounded word difference), but you may need
# to extend your FSA if it doesn't accept enough pairs. 
alph = {'a', 'b', 'c'}
transitions = {'a': [1, 2, 0], 'b': [0, 1, 2], 'c': [0, 1, 2]}
mod3a0 = FSA.FSA(3, {0}, alph, transitions )
order_automaton = orderAutomata.regular_split_shortlex(mod3a0, {0: ['a', 'b', 'c'], 1: ['b', 'c', 'a'], 2: ['c', 'a', 'b']})
# You can define your own ordering function here, but if you're using an FSA 
# as the basis for your ordering, you can leave the next line alone.
# An example of using a custom ordering function is given in BS12pKB.py.
Cox333.ordering = orderAutomata.make_ordering(alph, order_automaton)

# Now we can actually call pKB! I haven't used the arguments `max_rule_number` 
# or `max_rule_length` in this example, and `max_time` is frankly way larger
# than it needs to be for this.
# Note that these stopping conditions are soft, in that PKB will finish its
# current round of major steps before checking stopping conditions.
# So if you set `max_time` to 2, this might actually run for 4 seconds.
pKB(Cox333, max_time = 2000)

# (Note that without comments, you can actually write this in 35 lines.)
