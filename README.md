# data558_code
Polished Code Release for DATA558

This code implements linear SVM in a one-vs-one fashion by taking the mode across all classes
from a pair-wise huberized hinge loss function for classification purposes.

There are three .py files in this repo:

1. ovo_linear_svm_simulate.py

This module demostrates the functions on a simulated dataset and visualizes the objective function
as well as prints the performance metric.

There is no need to download data for this module, simply download .py and
```
python ovo_linear_svm_simulate.py
```

2. ovo_linear_svm_real_world.py.py

This module demostrates the functions on Image net dataset with extracted features while also 
visualizes the objective function as well as prints the performance metric.

In order to run this module, first download data from the data folder and
```
python ovo_linear_svm_real_world.py
```

Note: There will be a pop up figure for objective value curves.

3. comparison_against_sklearn.py

This module allows users to compare performance of the ovo linear svm function against sklearn's
version of SVC using linear kernel and compare the performance metrics.

In order to run this module, first download data from the data folder and
```
python comparison_against_sklearn.py
```


