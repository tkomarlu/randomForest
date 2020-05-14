#!/usr/bin/env python
import sys, os, os.path
import json

if __name__ == "__main__":

    from randomForest import run_train_test

    train_data = json.load(open('train.json'))
    dev_data = json.load(open('dev.json'))

    # accuracy = 0
    # while accuracy < 0.7:
    prediction = run_train_test(train_data['data'], train_data['label'], dev_data['data'])

    accuracy = len([i for i in range(len(prediction)) if prediction[i] == dev_data['label'][i]]) / float(len(prediction))
    '''
    TP = len([i for i in range(len(prediction)) if prediction[i] == dev_data['label'][i] and dev_data['label'][i] > 0])
    FP = len([i for i in range(len(prediction)) if prediction[i] == dev_data['label'][i] and dev_data['label'][i] > 0])
    all_pos = len([_ for _ in dev_data['label'] if _ > 0])
    if float(TP+FP) > 0:
        precision = TP / float(TP+FP)
    else:
        precision = 0
    recall = TP / float(all_pos)
    if precision + recall > 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0
    print(f1)
    '''
    print(accuracy)
