import functools
import warnings
import sklearn.metrics as skmetrics



def _handle_cropped(y_p):
    """
    A straightforward helper that simply averages multiple crops if they are present.

    Parameters
    ----------
    y_p: np.ndarray
         The predicted values with shape batch x targets (x <optional crops>)

    Returns
    -------
    y_p_mean: np.ndarray
              If there is an additional crop dimensions, mean across this dimension
    """
    if len(y_p.shape) == 2:
        return y_p
    elif len(y_p.shape) == 3:
        return y_p.mean(-1)
    else:
        raise ValueError("Predictions should be 1 or 2 dimensions in shape (excluding batches)")


def _binarize_two_class(y_p):
    if y_p.shape[-1] == 2:
        return y_p[..., -1]
    elif y_p.shape[-1] > 2:
        print("This simple metric implementation doesn't support multi-class targets.")
        return 0


def _get_prediction(outputs):
    """Checks if multiple outputs were provided, and selects"""
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def dn3_sklearn_metric(func):
    @functools.wraps(func)
    def wrapper(inputs, outputs, **kwargs):
        outputs = _get_prediction(outputs)
        y_p = _handle_cropped(outputs.detach().cpu().numpy()).argmax(-1)
        y_t = inputs[-1].detach().cpu().numpy()
        # Get all sorts of warning during training because batches aren't stable, we ignore these
        # careful because this could make debugging real problems in val/test impossible
        # TODO have some sort of warning system for the library to not do this when debugging...
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(y_t, y_p, **kwargs)
    return wrapper


def dn3_sklearn_binarized(func):
    @functools.wraps(func)
    def wrapper(y_t, y_p, **kwargs):
        y_p = _get_prediction(y_p)
        y_p = _binarize_two_class(y_p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(y_t[-1].detach().cpu().numpy(), y_p.detach().cpu().numpy(), **kwargs)
    return wrapper


@dn3_sklearn_binarized
def auroc(y_t, y_p):
    return skmetrics.roc_auc_score(y_t, y_p)


@dn3_sklearn_metric
def balanced_accuracy(y_t, y_p):
    return skmetrics.balanced_accuracy_score(y_t, y_p)
