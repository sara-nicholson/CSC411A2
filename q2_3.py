'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    #split the data based on classifcation
    ind = np.argsort(train_labels)
    sorted_labels = train_labels[ind]
    split_data = np.split(train_data[ind], np.where(np.diff(sorted_labels[:]))[0]+1)
    #calcualte eta, with ~Beta(2,2)
    alpha= 2
    beta = 2
    for k in range(10):
        Nc = np.count_nonzero(split_data[k],axis=0)
        N = len(split_data[k])
        eta[k] = (Nc+alpha-1)/(N+alpha+beta-2)
        
    return eta
   
def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    #reshape the data
    reshaped = []
    for i in range(len(class_images)):
        img_i = class_images[i]
        reshaped.append(np.reshape(img_i,(8,8)))
    #plot the reshaped data
    all_concat = np.concatenate(reshaped, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    #generate a new data point for each class, using a random binomial and eta
    for i in range(len(eta)):
        generated_data[i] = np.random.binomial(1,eta[i])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    likelihood = np.zeros((10,len(bin_digits)))
    #computer the generative likelihood
    for k in range(len(eta)):
        first = eta[k]**bin_digits
        second = (1-eta[k])**(1-bin_digits)
        likelihood[k] = np.prod(first*second,axis=1)
            
    return np.log(likelihood.T)

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    #get the generative likelyhood
    gen_likelihood = generative_likelihood(bin_digits,eta)
    #get the p(x)
    p_x = gen_likelihood*(np.log(1/10))
    #calculate the conditional log likelihood
    CLL = gen_likelihood + np.log(1/10) - p_x
    
    return CLL

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    #csplit the conditional likelihoods based on labels
    ind = np.argsort(labels)
    sorted_labels = labels[ind]
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    split_likelihood = np.split(cond_likelihood[ind], np.where(np.diff(sorted_labels[:]))[0]+1)
    #calculte the average conditional likelihood based on true class
    avg = np.zeros(len(split_likelihood))
    for i in range(len(split_likelihood)):
        avg[i] = np.mean(split_likelihood[i][:][i])

    return avg

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    labels = np.argmax(cond_likelihood,axis=1)
    return labels

def classification_accuracy(calc_label,true_label):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    #determine accuracy of the labeling
    size_data = np.size(calc_label)
    accuracy = 1- (np.count_nonzero(true_label - calc_label)/size_data)
    return accuracy*100
    
def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    CLL_train = conditional_likelihood(train_data,eta)
    CLL_test = conditional_likelihood(test_data,eta)
    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    avg_train = avg_conditional_likelihood(train_data,train_labels,eta)
    print("avg CLL train data: ",avg_train)
    avg_test = avg_conditional_likelihood(test_data,test_labels,eta)
    print("avg CLL test data: ",avg_test)
    
    classify_train = classify_data(train_data,eta)
    classify_test = classify_data(test_data,eta)
    
    train_accuracy = classification_accuracy(classify_train,train_labels)
    print("Train accuracy: ",train_accuracy)
    test_accuracy = classification_accuracy(classify_test,test_labels)
    print("Test accuracy: ",test_accuracy)
    #generate_new_data(eta)
    

if __name__ == '__main__':
    main()
