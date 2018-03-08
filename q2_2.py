'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    #split the data based on the labels they are classified as
    ind = np.argsort(train_labels)
    sorted_labels = train_labels[ind]
    split_data = np.split(train_data[ind], np.where(np.diff(sorted_labels[:]))[0]+1)
    #compute the means for each class
    for i in range(len(split_data)):
        means[i,:] = np.mean(split_data[i],axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    #split the data based on the labels they relate to
    ind = np.argsort(train_labels)
    sorted_labels = train_labels[ind]
    split_data = np.split(train_data[ind], np.where(np.diff(sorted_labels[:]))[0]+1)
    #compute the covariance matricies for each of the classes
    for i in range(len(split_data)):
        for j in range(len(train_data[1])):
            for k in range(len(train_data[1])):
                left = ([x[j] for x in split_data[i]] - np.mean([x[j] for x in split_data[i]]))
                right = ([x[k] for x in split_data[i]] - np.mean([x[k] for x in split_data[i]]))
                covariances[i][j][k] = (1/len(left))*sum(left*right)
#    
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    reshaped = []
    #reshaped the diagonals
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        reshaped.append(np.reshape(cov_diag,(8,8)))
    #plot all the data together
    all_concat = np.concatenate(reshaped, 1)
    plt.imshow(np.log(all_concat), cmap='gray')
    plt.show()
def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    likelihood = np.zeros((len(digits),len(means)))
    #determine the geenrative log likelihood
    for i in range(len(digits)):
        for j in range(len(means)):
            inv_cov = np.linalg.inv(covariances[j])
            numer = ((-1/2)* np.dot(np.dot((digits[i] - means[j]).T,(inv_cov)),(digits[i] - means[j])))
            det_cov = np.linalg.det(covariances[j])
            denom = np.log(np.sqrt(np.power(np.pi*2,8/2))*det_cov)
            likelihood[i][j] = numer - denom
        
    return likelihood
     
def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    #determine the generative likelihood
    gen_likelihood = generative_likelihood(digits,means,covariances)
    #determine the probability of x using the 
    p_x = gen_likelihood*(np.log(1/10))
    con_likelihood = gen_likelihood + np.log(1/10) - p_x
    
    return con_likelihood



def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    ind = np.argsort(labels)
    sorted_labels = labels[ind]
    cond_likelihood = conditional_likelihood(digits, means,covariances)
    split_likelihood = np.split(cond_likelihood[ind], np.where(np.diff(sorted_labels[:]))[0]+1)
    avg = np.zeros(len(split_likelihood))
    for i in range(len(split_likelihood)):
        avg[i] = np.mean(split_likelihood[i][:][i])
    
    # Compute as described above and return
    return avg

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    labels = np.argmax(cond_likelihood,axis=1)
    return labels

def classification_accuracy(calc_label,true_label):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    #determine accuracy of classification
    size_data = np.size(calc_label)
    accuracy = 1- (np.count_nonzero(true_label - calc_label)/size_data)
    return accuracy*100

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    train_means = compute_mean_mles(train_data, train_labels)
    train_covariances = compute_sigma_mles(train_data, train_labels)
    
    plot_cov_diagonal(train_covariances)
    
    train_averages = avg_conditional_likelihood(train_data,train_labels,train_means,train_covariances)
    test_averages = avg_conditional_likelihood(test_data,test_labels,train_means,train_covariances)

    detem_labels_train = classify_data(train_data,train_means,train_covariances)
    detem_labels_test = classify_data(test_data,train_means,train_covariances)

    train_accuracy = classification_accuracy(detem_labels_train,train_labels)
    test_accuracy =classification_accuracy(detem_labels_test,test_labels)
    print(train_accuracy)
    print(test_accuracy)

if __name__ == '__main__':
    main()
    