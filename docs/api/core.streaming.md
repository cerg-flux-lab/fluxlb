# `fluxlb.core.streaming`

The streaming step: a push of every post-collision population one lattice spacing along its
own velocity, periodic by construction, as one `torch.roll` per direction. It is a fixed
permutation of the population array and the half of the update that sits outside the
collision seam. For the collide-stream loop and why the seam is drawn there, see the user
guide.

```{eval-rst}
.. automodule:: fluxlb.core.streaming
   :members:
```
