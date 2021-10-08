"""
This file contains helpful functions for implementing a gradient
descent algorithm.

TODO:
    Solve normal equations and test against these

Alex Angus, Michael Robertson

September 6, 2019
"""
import pandas as pd
import numpy as np
import scipy.linalg
import random
import sys
import copy, time

def read_wine_csv(filename):
    """
    This function takes a csv file as a parameter and returns a
    dictionary with feature names as keys and numpy arrays as values
    """
    data = pd.read_csv(filename)
    feature_name_list = data.columns[0].split(';')
    data_length = len(data)
    feature_data = {}

    for feature in feature_name_list:
        feature_data.update({feature.replace('"', '') : np.zeros(data_length)})

    for row in range(data_length):
        feature_values = data.iloc[row, 0].split(';')
        feature_data['fixed acidity'][row] = feature_values[0]
        feature_data['volatile acidity'][row] = feature_values[1]
        feature_data['citric acid'][row] = feature_values[2]
        feature_data['residual sugar'][row] = feature_values[3]
        feature_data['chlorides'][row] = feature_values[4]
        feature_data['free sulfur dioxide'][row] = feature_values[5]
        feature_data['total sulfur dioxide'][row] = feature_values[6]
        feature_data['density'][row] = feature_values[7]
        feature_data['pH'][row] = feature_values[8]
        feature_data['sulphates'][row] = feature_values[9]
        feature_data['alcohol'][row] = feature_values[10]
        feature_data['quality'][row] = feature_values[11]

    return feature_data


def data_status(feature_dictionary):
    """
    This funciton checks the data for nan values.
    """
    nans = 0
    for feature in feature_dictionary.keys():
        feature_array = feature_dictionary[feature]
        if np.isnan(feature_array.any()):
            nans += 1
    print("There are", nans, "features that have nan values.")


def normalize_data(feature_dictionary):
    """
    This function normalizes the data. It returns a normalized dictionary where
    the mean of each feature is 0, and the standard deviation of each feature
    is 1.
    """
    normalized_feature_dictionary = copy.copy(feature_dictionary)
    for feature in feature_dictionary.keys():
        if feature == 'quality':
            continue
        value_array = feature_dictionary[feature]
        feature_mean = sum(value_array) / len(value_array)
        std_sum = sum((value_array - feature_mean)**2)
        feature_std = np.sqrt(std_sum / len(value_array))
        normalized_feature_dictionary[feature] = (value_array - feature_mean) / feature_std

    return normalized_feature_dictionary
    
    
def dictionary_to_array(feature_dictionary):
    """
    takes the feature dictionary and returns a 2-d array of features
    """
    feature_array = []
    for feature in feature_dictionary.keys():
        if feature == 'quality':
            continue
        feature_array.append(list(feature_dictionary[feature]))
    return np.array(feature_array)


def shuffle(feature_dictionary, seed = 10):
    """
    shuffle all features in place with the same seed to ensure mapping of different features to the same quality value.
    """
    for feature in feature_dictionary.keys():
        random_state = np.random.RandomState(seed)
        random_state.shuffle(feature_dictionary[feature])
    return feature_dictionary



def add_second_order_terms(feature_dictionary, keys):
    """
    adds a second order term to each feauture in keys. This new second order term will have the key:

    [feature]2
    """
    return_copy = copy.copy(feature_dictionary)
    if type(keys) == list:
        for feature in keys:
            if feature == 'quality': ##shouldn't happen
                continue
            return_copy.update({feature + "2" : np.square(feature_dictionary[feature])})
    else:
        return_copy.update({keys + "2" : np.square(feature_dictionary[keys])})
    return return_copy

def train_validate_test(feature_dictionary):
    """
    returns a train, validate, and test dictionary.
    """
    train_dictionary = {}
    validate_dictionary = {}
    test_dictionary = {}
    for feature in feature_dictionary.keys():
        feature_array = feature_dictionary[feature]
        array_length = len(feature_array)
        split_feature_array = np.split(feature_array, [int(array_length * 0.8), # 80% train
                                                       int(array_length * 0.95), # 15% validate
                                                       array_length]) # 5% test

        train_dictionary.update({feature : split_feature_array[0]}) #assign each segment to new feature dictionary
        validate_dictionary.update({feature : split_feature_array[1]})
        test_dictionary.update({feature : split_feature_array[2]})
    """
    print('len(train_dictionary)', len(train_dictionary))
    print('len(train_dictionary)[\'quality\']', len(train_dictionary['quality']))
    """

    return train_dictionary, validate_dictionary, test_dictionary

def initial_weights(shape):
    """
    returns an array of random weights
    shape is the number of weights
    """
    return np.random.rand(shape)

def model(theta, feature_values):
    """
    our model function of the form:

        y_hat = theta1 * x1 + theta2 * x2 + theta3 * x3 + ...

    where xn is an array of feature values and each theta represents that
    feature's contribution to the wine quality. 
    """
    if type(feature_values) != np.ndarray:
        feature_data_array = np.empty(len(feature_values) - 1, dtype=np.ndarray)
        feature_iterator = 0
        for feature in feature_values.keys():
            if feature == 'quality':
                continue
            feature_data_array[feature_iterator] = feature_values[feature]
            feature_iterator += 1
        feature_values = feature_data_array
    model_value = 0
    for i in range(len(feature_values)):
        model_value += theta[i] * feature_values[i]
    return model_value
    
    
def ridge_model(theta, feature_values, regularization):
    """
    our model function with ridge regression:

        y_hat = theta1 * x1 + theta2 * x2 + theta3 * x3 + ... 
                                    + lambda * [theta1^2 + theta2^2 + ... ]

    where xn is an array of feature values and each theta represents that
    feature's contribution to the wine quality. lambda is the regularization
    hyperparameter.
    """
    if type(feature_values) != np.ndarray:
        feature_data_array = np.empty(len(feature_values) - 1, dtype=np.ndarray)
        feature_iterator = 0
        for feature in feature_values.keys():
            if feature == 'quality':
                continue
            feature_data_array[feature_iterator] = feature_values[feature]
            feature_iterator += 1
        feature_values = feature_data_array
    model_value = 0
    for i in range(len(feature_values)):
        model_value += theta[i] * feature_values[i] 
    model_value += regularization * np.sum(np.square(theta))
    return model_value
    

def normal_eqs(feature_dictionary):
    """
    Trusting scipy.linalg.orth to orthogonalize, somehow this increases the MSE
    """
    mutable_dict = copy.copy(feature_dictionary)
    y = np.array(mutable_dict.pop('quality'))
    #to orthogonalize, add scipy.linalg.orth
    a = np.transpose(np.array(list(mutable_dict.values())))
    #solve (X^TX)^-1X^T y
    theta =np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(a),a)), \
    np.transpose(a)),y)
    return theta


#calculate the mean squared error for our weights
def MSE(theta, feature_dictionary, model_type = 'linear', regularization = None):
    if((model_type != 'linear') and (regularization == None)):
        print("Specify regularization hyperparameter.")
        return
    feature_length = len(feature_dictionary.keys()) - 1
    feature_data_array = np.empty(feature_length, dtype=np.ndarray)
    feature_iterator = 0
    for feature in feature_dictionary.keys():
        if feature == 'quality':
            continue
        feature_data_array[feature_iterator] = feature_dictionary[feature]
        feature_iterator += 1
    if model_type == 'linear':
        y_hat = model(theta, feature_data_array)
    elif model_type == 'ridge':
        y_hat = ridge_model(theta, feature_data_array, regularization)
    return np.linalg.norm(y_hat - feature_dictionary['quality']) / len(feature_dictionary['quality'])


def descent(feature_dictionary, threshold, alpha, model_type = 'linear', 
            regularization = None):
    """
    Performs gradient descent.
    
    inputs:
        feature_dictionary is a dictionary containing feature values
    
        threshold is our stopping tolerance (difference between successive theta updates)
    
        alpha is the learning rate
    
        model_type is the keyword for type of model we're using. 'linear' is no regularization  
                                                                 'ridge' is ridge regression regularization
    
    output:
        theta is an array of trained weights
    """
    starttime = time.time()
    feature_length = len(feature_dictionary.keys()) - 1 #minus 1 because quality isn't a feature

    quality_data = feature_dictionary['quality']
    
    if((model_type != 'linear') and (regularization == None)):
        print("Specify regularization hyperparameter.")
        return
        
    #feature dictionary --> feature array for defined iterations
    feature_data_array = np.empty(feature_length, dtype=np.ndarray)
    feature_iterator = 0
    for feature in feature_dictionary.keys():
        if feature == 'quality':
            continue
        feature_data_array[feature_iterator] = feature_dictionary[feature]
        feature_iterator += 1

    theta_prediction = initial_weights(feature_length) #initial weights
    theta = np.copy(theta_prediction)
    count = 0
    error = 1.0
    while abs(error) > threshold:
        if model_type == 'linear':
            y_hat = model(theta, feature_data_array)
        elif model_type == 'ridge':
            y_hat = ridge_model(theta, feature_data_array, regularization)
        for feature in range(feature_length):
            """
            This is where we calculate the value of our loss function:
                sum( (y_hat - y)x ) = 0
            """
            sum_value = np.sum((y_hat - quality_data) * feature_data_array[feature])
            #then update weights
            theta[feature] = theta_prediction[feature] - alpha * sum_value

        error = sum(theta - theta_prediction)
        theta_prediction = np.copy(theta)
        count += 1
    if count%50 == 0:
        print(error)

    """
    print('Total time to regress dataset of size {} by {} by gradient descent: {:3.2f} ms'\
            .format(len(feature_data_array),len(feature_data_array[0]),(time.time() - starttime)*1000))
    """
    return theta #return the fitted linear weights
    
def make_one_prediction(feature_values, trained_theta):
    """
    returns a list of individual quality predictions based on the 
    trained weights.
    """
    prediction_list = []
    features_list = []
    feature_data_list = []
    for feature in feature_values.keys():
        if feature == 'quality':
            continue
        feature_data_list.append(list(feature_values[feature]))
    feature_data_array = np.array(feature_data_list)
    for wine in range(len(feature_data_array[0])):
        feature_values = np.array(len(feature_data_array))
        for feature in range(len(feature_data_array)):
            print(np.shape(feature_data_array))
            feature_values[feature] = feature_data_array[feature][wine]
        feature_list.append(feature_values)
        prediction_list.append(model(trained_theta, feature_values))
        
    return features_list, prediction_list

def hyper_parameter_search(feature_dictionary, alpha = 0.0000002, tol =0.00001):
    """
    Add comparison to normal equation mse for each

    TODO: Save weights, compare validate MSE
        Investigate which combination of features is best using normal equations
    """
    normalized_shuffled_features = shuffle(normalize_data(feature_dictionary)) #shuffled as well

    #find the linear normal solutions
    mse_normal_dict ={}
    normal_weights_dict ={}
    normal_weights = normal_eqs(normalized_shuffled_features)
    normal_weights_dict.update({'linear' : normal_weights})
    mse_normal_dict.update({'linear' : MSE(normal_weights, feature_dictionary)})

    mse_sgd_dict = {}
    sgd_weights_dict={}
    (train, validate, _) = train_validate_test(normalized_shuffled_features)
    weights = descent(train, tol, alpha) #linear
    sgd_weights_dict.update({'linear' : weights})
    mse_sgd_dict.update({"linear" : MSE(weights, validate)})

    #add one second order term
    for i in range(len(normalized_shuffled_features) - 1):
        enhanced_feature_dictionary = {}
        starttime = time.time()
        feature_to_square = list(feature_dictionary.keys())[i]
        enhanced_feature_dictionary = add_second_order_terms(normalized_shuffled_features, feature_to_square)
        (train, validate, _) = train_validate_test(enhanced_feature_dictionary)

        #normal equations
        weights = normal_eqs(enhanced_feature_dictionary)
        print('Solved normal equation of size {} by {} for {} in {:3.2f} ms'\
                .format(len(enhanced_feature_dictionary)-1,\
                len(enhanced_feature_dictionary['quality'])-1,\
                feature_to_square, (time.time() - starttime)*1000))
        normal_weights_dict.update({feature_to_square + '^2': weights})
        mse_normal_dict.update({feature_to_square:MSE(weights, validate)})

        #sgd
        weights = descent(train, tol, alpha)
        mse_sgd_dict.update({feature_to_square : MSE(weights, validate)})
        print('Trained with {} learning rate with {}^2 as an additional feature '
                'in {:3.2f} ms\n'\
                .format(alpha, feature_to_square, time.time() - starttime))
        sgd_weights_dict.update({feature_to_square : weights})


    mse_sgd_list = [(v,k) for k, v in mse_sgd_dict.items()]
    mse_sgd_list.sort()
    mse_normal_list = [(v,k) for k, v in mse_normal_dict.items()]
    mse_normal_list.sort()
        ##train and find the mse on the validation set
    for i in range(len(mse_normal_list)):
        print('Normal mean squared error for {}: {} \n'.format(\
                    mse_normal_list[i][1], mse_normal_list[i][0], '\n'))
        print('Trained mean squared error for {}: {} \n'.format(\
                    mse_sgd_list[i][1], mse_sgd_list[i][0], '\n'))

    print('Best normal mse comes from squaring feature {}'
            ' giving mse {}'.format(mse_normal_list[0][1], mse_normal_list[0][0]))
    print('Best sgd mse comes from squaring feature {}'
            ' giving mse {}'.format(mse_sgd_list[0][1], mse_sgd_list[0][0]))

    #return best weights

    return (mse_normal_list[0][1], normal_weights_dict[mse_normal_list[0][1]]), (mse_sgd_list[0][1],sgd_weights_dict[mse_sgd_list[0][1]])
    
    
def bias(theta, feature_dictionary, model_type = 'linear', regularization = None):
    """
    returns the bias of a given model. Bias is defined as:
    
        1/n * sum( y_hat - y)
    """
    if((model_type != 'linear') and (regularization == None)):
        print("Specify regularization hyperparameter.")
        return
    if model_type == 'linear':
        y_hat = model(theta, feature_dictionary)
    elif model_type == 'ridge':
        y_hat = ridge_model(theta, feature_dictionary, regularization)
    y = feature_dictionary['quality']
    return np.sum(y_hat - y) / len(y)


def main():
    """
    argument keys:
        1 : features file
        2 : threshold (stopping parameter)
        3 : learning parameter (alpha)_
    """
    feature_dictionary = read_wine_csv(sys.argv[1])
    # normal_eqs(feature_dictionary)
    # normalized_features = normalize_data(feature_dictionary)
    # normalized_features = feature_dictionary
    # trained_weights = descent(normalized_features, float(sys.argv[2]),  0.0000002)
    # normal_weights = normal_eqs(feature_dictionary)
    # print('weights:', trained_weights)
    # print(MSE(trained_weights, feature_dictionary))
    # print('normal weights:', normal_weights)
    # print(MSE(normal_weights, feature_dictionary))
    print(hyper_parameter_search(feature_dictionary))


if __name__ == "__main__":
    main()
