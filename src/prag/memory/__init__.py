"""Memory: session and long-term tiers, with the namespace boundary enforced at the store.

Memory enters context in its own ``[MEMORY]`` region with ``[M]`` markers, never as evidence.
``MemoryItem`` and ``EvidenceGroup`` are distinct types with no conversion path, and the citation
validator refuses cross-namespace references, so a thing the user said can never be cited as a
thing a document said.

Imports only ``core``.
"""

from prag.memory.stores import PrincipalMemoryStore, decisions_in, entities_in

__all__ = ["PrincipalMemoryStore", "decisions_in", "entities_in"]
