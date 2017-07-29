import helper
import submission

CLASSIFIER_PATH = './classifier_file'

if __name__ == '__main__':
    training_data = helper.read_data('./asset/training_data.txt')
    testing_data = helper.read_data('./asset/tiny_test.txt')

    submission.train(training_data, CLASSIFIER_PATH)
    predict_res = submission.test(testing_data, CLASSIFIER_PATH)
    
    print ('testing_data: ',testing_data)
    print ('predicting_res: ', predict_res)