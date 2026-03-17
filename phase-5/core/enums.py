from enum import Enum

class MemoryType(Enum):
    RAW = 0b0001 #1
    SEMANTIC = 0b0010 #2
    EPISODIC = 0b0100 #4
    PROCEDURAL = 0b1000 #8



def get_memory_type_human(memory_type: int) -> list[str]:
    """
    With &, it actually checks if that specific bit is set in the bitmask 
    that's the correct way to decode a bitmask.

    if memory_type = 5 (0101)
    1 & 5 = true (since 1 = 0001)
    2 & 5 = false (2 = 0010)
    4 & 5 = true (4 = 0100)
    returns ["raw", "episodic"]
    """
    return [m.name.lower() for m in MemoryType if memory_type & m.value]


def calculate_memory_type(memory_typename_list: list[str]) -> int:
    type_value_map = {m.name.lower(): m.value for m in MemoryType}
    bits = 0
    for name in memory_typename_list:
        if name in type_value_map:
            bits |= type_value_map[name]
    return bits