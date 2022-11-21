import torch
import numpy as np
from scipy.linalg import sqrtm


class Preprocessor:
    """
    Base class for various preprocessing actions. Sub-classes are called with a subclass of `_Recording`
    and operate on these instances in-place.

    Any modifications to data specifically should be implemented through a subclass of :any:`BaseTransform`, and
    returned by the method :meth:`get_transform()`
    """
    def __call__(self, recording, **kwargs):
        """
        Preprocess a particular recording. This is allowed to modify aspects of the recording in-place, but is not
        strictly advised.

        Parameters
        ----------
        recording :
        kwargs : dict
                 New :any:`_Recording` subclasses may need to provide additional arguments. This is here for support of
                 this.
        """
        raise NotImplementedError()

    def get_transform(self):
        """
        Generate and return any transform associated with this preprocessor. Should be used after applying this
        to a dataset, i.e. through :meth:`preprocess`

        Returns
        -------
        transform : BaseTransform
        """
        raise NotImplementedError()
