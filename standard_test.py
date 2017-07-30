import helper
import submission
from sklearn.metrics import f1_score

if __name__ == '__main__':
    training_data = helper.read_data('./asset/training_data.txt')
    classifier_path = './asset/classifier.dat'
    submission.train(training_data, classifier_path)

    test_data = helper.read_data('./asset/tiny_test.txt')
    prediction = submission.test(test_data, classifier_path)
    print(prediction)

    ground_truth = [1, 1, 2, 1]
    print(f1_score(ground_truth, prediction, average='micro'))