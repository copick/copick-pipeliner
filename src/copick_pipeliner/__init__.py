"""copick-pipeliner: ccpem-pipeliner job types for copick-based particle picking.

Five job types are registered through the ``ccpem_pipeliner.jobs`` entry-point group
(see ``pyproject.toml``), the same mechanism ``zarr-particle-tools`` uses for its
``zarrparticletools.*`` jobs:

``copick.project``      build a copick project from a RELION ``tomograms.star`` or a
                        cryoET Data Portal dataset mirror
``copick.portalpicks``  import deposited portal annotations as picks and export them
``copick.easymode``     copick-easymode segmentation -> picks -> ``particles.star``
``copick.boundary``     octopi ``tomogram-boundary`` cleanup of an upstream pick set
``copick.membrain``     MemBrain-seg membrane segmentation (copick-torch)

The job classes import only ``pipeliner`` and the standard library, so they load in the
control process without copick, TensorFlow or torch. All scientific work happens in the
``copick-pipeliner-tools`` CLI (``copick_pipeliner.tools``), which runs in the picking
environment and shells out to ``copick`` / ``octopi``.
"""

__version__ = "0.1.11"
