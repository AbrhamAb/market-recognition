from typing import Any, Tuple, List, Sequence


def precision_recall_fscore_support(y_true: Sequence[int], y_pred: Sequence[int], *args: Any,
                                    **kwargs: Any) -> Tuple[List[float], List[float], List[float], List[int]]: ...


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], *args: Any, **kwargs: Any) -> str: ...


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                     *args: Any, **kwargs: Any) -> List[List[int]]: ...
