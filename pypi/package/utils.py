import numpy as np
class utils():
  def _release_memory():
        mempool = np.get_default_memory_pool()
        pinned_mempool = np.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()