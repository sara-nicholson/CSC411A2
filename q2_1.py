'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        #train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distances = self.l2_distance(test_point)
        #determine the indicies of the k smallest distances
        kIndicies = np.argpartition(distances,k)[:k]
        labels = self.train_labels[kIndicies]
        (value,count) = np.unique(labels,return_counts=True)
        ind = np.argmax(count)
        digit = value[ind]
        #print(digit, value,count)
        return digit

def cross_validation(knn, k_range=np.arange(1,16)):
    """
    Perform k-fold cross validation in the range of k_range 
    using 10 splits 
    
    """

   
    kf = KFold(n_splits = 10)
    accuracies = []
    data = knn.train_data
    for k in k_range:
        kth_accuracies = []
        for train,test in kf.split(data):
            kFoldKNN = KNearestNeighbor(data[train], knn.train_labels[train])
            testLabels = []
            test_data = data[test]
            for i in range(len(test_data)):
                testLabels.append(kFoldKNN.query_knn(test_data[i],k))
            accuracy = classification_accuracy(kFoldKNN,k,testLabels, knn.train_labels[test])
            kth_accuracies.append(accuracy)
        accuracies.append((kth_accuracies))
        
    return accuracies

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    size_data = np.size(eval_data)
    accuracy = 1- (np.count_nonzero(eval_data - eval_labels))/size_data
    return accuracy*100

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)
     
    trainLabel_1 = []
    for i in range(len(train_data)):
        trainLabel_1.append(knn.query_knn(train_data[i], 1))
    
    
    trainLabel_15 = []
    for j in range(len(train_data)):
        trainLabel_15.append(knn.query_knn(train_data[j],15))
    
    testLabel_1 = []
    for i in range(len(test_data)):
        testLabel_1.append(knn.query_knn(test_data[i], 1))
    
    testLabel_15 = []
    for j in range(len(test_data)):
        testLabel_15.append(knn.query_knn(test_data[j],15))
        
    trainAccuracy_1 = classification_accuracy(knn,1,trainLabel_1,train_labels)
    trainAccuracy_15 = classification_accuracy(knn,15,trainLabel_15, train_labels)
    testAccuracy_1 = classification_accuracy(knn,1,testLabel_1,test_labels)
    testAccuracy_15 = classification_accuracy(knn,15,testLabel_15, test_labels)
    print("Train Accuracy, k = 1: ",trainAccuracy_1)
    print("Train Accuracy, k = 15: ", trainAccuracy_15)
    print("Test Accuracy, k = 1: ",testAccuracy_1)
    print("Test Accuracy, k = 15: ",testAccuracy_15)
    
    #do kfold cross validation on the data set  and output the k that gives the max accuracy
    accuracies = cross_validation(knn)

    #find max average accuracies (index +1 is the k used)
    averages = []
    for x in range(len(accuracies)):
        averages.append(np.mean(accuracies[x]))
    print(averages)   
    index = np.argmax(averages)
    max_k = index+1
    max_accuracies = accuracies[index]
    
    print("The max k is: {} with accuracies of {}".format(max_k,max_accuracies))
    

if __name__ == '__main__':
    main()