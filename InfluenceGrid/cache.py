from functools import wraps
from collections.abc import Iterable


# from burnysc2
def cache_once_per_frame(f):
    """This decorator caches the return value for one game loop,
    then clears it if it is accessed in a different game loop.
    Only works on properties of the bot object, because it requires
    access to self.state.game_loop"""

    @wraps(f)
    def inner(self, *args, **kwargs):
        property_cache = "_cache_" + f.__name__
        state_cache = "_frame_" + f.__name__
        cache_updated = getattr(self, state_cache, -1) == self._bot.state.game_loop
        if not cache_updated:
            print('updated', self._bot.state.game_loop, getattr(self, state_cache, -1))
            setattr(self, property_cache, f(self, *args, **kwargs))
            setattr(self, state_cache, 1)
        else:
            print('not updated')
        cache = getattr(self, property_cache)
        should_copy = callable(getattr(cache, "copy", None))
        if should_copy:
            return cache.copy()
        return cache
    return inner

