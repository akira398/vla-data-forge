"""
Abstract base class for dataset readers.

All dataset readers must implement this interface.  This enables the generation
pipeline and curation pipeline to treat any dataset uniformly — they only
depend on ``DatasetReader``, not on the concrete ECoT or Bridge implementation.

Design
------
- ``__iter__`` yields episodes lazily; callers that only need a few episodes do
  not pay the cost of loading the entire dataset.
- ``load_episode`` provides random access by ID when needed.
- ``episode_ids`` lists all available IDs (a cheap index-level call).
- Concrete readers declare ``dataset_name`` and ``schema_class`` as class
  attributes so callers can inspect metadata without instantiating a reader.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

# TypeVar for the episode type returned by this reader
E = TypeVar("E")


class DatasetReader(abc.ABC, Generic[E]):
    """
    Abstract base class for all dataset readers.

    Type parameter ``E`` is the episode dataclass returned by this reader,
    e.g. ``ECoTEpisode`` or ``BridgeEpisode``.

    Minimal concrete implementation::

        class MyReader(DatasetReader[MyEpisode]):
            dataset_name = "my_dataset"

            def __iter__(self) -> Iterator[MyEpisode]:
                for raw in self._source:
                    yield self._parse(raw)

            def load_episode(self, episode_id: str) -> Optional[MyEpisode]:
                ...

            def episode_ids(self) -> List[str]:
                ...
    """

    dataset_name: str = ""
    """Human-readable identifier for the dataset.  Set as a class attribute."""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def __iter__(self) -> Iterator[E]:
        """Yield episodes one at a time.  Should be re-entrant (recreatable)."""
        ...

    @abc.abstractmethod
    def load_episode(self, episode_id: str) -> Optional[E]:
        """
        Load a single episode by ID.  Returns None if not found.
        Implementations may raise on invalid IDs instead of returning None
        if that is more natural for the underlying storage format.
        """
        ...

    @abc.abstractmethod
    def episode_ids(self) -> List[str]:
        """
        Return all episode IDs available in this split.
        May trigger a full index scan; cache the result if called frequently.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers (override if needed)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of episodes.  Subclasses can override with a cheaper impl."""
        return len(self.episode_ids())

    def take(self, n: int) -> List[E]:
        """Return the first ``n`` episodes as a list."""
        episodes: List[E] = []
        for ep in self:
            episodes.append(ep)
            if len(episodes) >= n:
                break
        return episodes

    def info(self) -> Dict[str, Any]:
        """Return a human-readable summary of this reader."""
        return {
            "dataset_name": self.dataset_name,
            "num_episodes": len(self),
        }
