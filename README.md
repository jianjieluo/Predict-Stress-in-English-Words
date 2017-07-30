# Predict-Stress-in-English-Words
Machine Learning Project: Auto predict the place of stress in a word.

A project task of a summer one-week course Data Mining in Sun Yat-sen University, China.

More project requirements are [here](https://github.com/longjj/Predict-Stress-in-English-Words/blob/master/Project.ipynb).

## Usage

```bash
git clone https://github.com/longjj/Predict-Stress-in-English-Words
cd Predict-Stress-in-English-Words
```

### Run Standard Test

**You need to use python3. The experiment environment is [anaconda: python3.6](https://www.continuum.io/downloads)**.

You can also refer to these page to set up [my experiment environment](https://github.com/longjj/sysu-dm-summer/blob/master/0.self-evaluation/L0.python3-and-jupyter.ipynb).


```python
python standard_test.py
```

This script is a top module to test our team submission predict result.

### Run Random test


```python
python random_test.py
```

This script is a top module for us to test our classifier module.
Since the standard test set is too small, we use `cross_validation` here to split the standard big training data set into training set and testing set to check the `f1`.

You can also **use it to test your own change in the `submission.py`**

### Related files

1. `submission.py`: Our submitted code. Our major training and testing code. The most importance things for further learning are that you just need to change the related code in `getInfoOfPronsFromTrain()` and `getInfoFromTest()` to **generate your wanted features matrix and labels array**, and modify the `get_selected_classifier()` method to **choose a certain classifier you want**. Then you can run `random_test` to check your output.

2. `/doc`: there are all the related learning note of our 3 team members. It may be a little mass. The project report is also in it.
3. `helper.py`: a functional tool module provided from our tutor to read the testing and training data.

## Original Contributors

It is just a course project at the beginning.
The other team members in our team are:
1. [mgsweet](https://github.com/mgsweet)
2. [LebronX](https://github.com/LebronX)

other contribution are welcomed.

## Others

1. Some docs may be written in Chinese. In code files I **use mostly comments in English. It is not difficult to understand the code without the Chinese comments.**

2. Future dev: update the related docs and comments in English.
3. I hope this repo can give **a reliable reference to those who are new in data mining.**
4. [Related course resources](https://github.com/longjj/sysu-dm-summer)
