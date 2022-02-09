import matplotlib.pyplot as plt # 3.1.2    #Required libraries and versions.
import numpy as np # 1.18.0
import pandas as pd # 0.25.3

#Python 3.6.9
global test_predictions
global test_outputs
global test_labels
global accuracy

def sigmoid(x):                            # Sigmoid function.
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):                 # Derivative of sigmoid function.
    return 0.005 * x * (1 - x)

def read_and_divide_into_train_and_test(csv_file):          # Function for reading csv file.
    global train_inputs, train_labels, test_inputs, test_labels
    df = pd.read_csv(csv_file)                              # Main read method.
    df = df.replace("?", np.NaN)                            # Replacing missing values with NaN.
    df.drop(["Code_number"],1,inplace=True)                 # Dropping unwanted column.
    df.fillna(df.median(),inplace=True)                     # Changing missing values(NaN) with median of dataset.
    df = df.astype(int)                                     # Changing type of the values of the dataset with integer.

    trains_tests = round(0.8 * df.shape[0])                           #
    train_inputs = df.iloc[:trains_tests, :9].values                  #
    train_labels = df.iloc[:trains_tests,9].values                    # Splitting dataset into two pieces.
    train_labels = train_labels.reshape(train_labels.shape[0], -1)    # Assigning train and test sets.
    test_inputs = df.iloc[trains_tests:, :9].values                   #
    test_labels = df.iloc[trains_tests:, 9].values                    #

    train_inputs = train_inputs.astype(float)     #
    test_inputs = test_inputs.astype(float)       # Converting type of values float
    test_labels = test_labels.astype(float)       # in order to do operations.
    train_labels = train_labels.astype(float)     #

    correl = df.iloc[:trains_tests, :9].corr()    # Calculating correlation of set
    correl_vals = correl.values                   # in order to plot heatmap.

    fig, ax = plt.subplots()
    im = ax.imshow(correl)                          # Plotting part
    for i in range(correl.shape[1]):                # of the heatmap.
        for j in range(correl.shape[1]):
            ax.text(j,i,round(correl_vals[i,j],2),ha="center", va="center", color="b")
    fig.tight_layout()
    plt.xticks(range(correl.shape[1]), correl.columns, fontsize=7, rotation=25)
    plt.yticks(range(correl.shape[1]), correl.columns, fontsize=7)
    ax.set_title("Pairwise Correlation Heatmap")
    plt.show()


    return train_inputs, train_labels, test_inputs, test_labels

def run_on_test_set(test_inputs, test_labels, weights):          # Function for running operations on test set.
    global accuracy
    tp = 0
    total_sample = 0
    test_predictions = sigmoid(np.dot(test_inputs, weights))
    test_outputs = test_labels
    test_predictions = np.round(test_predictions,0)              # Making predictions (rounding 1 or 0)

    for predicted_val, label in zip(test_predictions, test_outputs):
        if predicted_val == label:
                tp += 1
        total_sample = total_sample + 1
    accuracy = tp / total_sample
    #print(accuracy)
    return accuracy                                              # Returns the accuracy in order to plot its change.



def plot_loss_accuracy(accuracy_array, training_loss):                      # Function for plotting the
    epochs = range(1,2501)                                                  # training loss and the
    plt.plot(epochs, training_loss, 'g', label='Training Loss')             # test accuracy values.
    plt.plot(epochs, accuracy_array, 'b', label='Test Accuracy')
    plt.title('Training Loss and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend(["Training Loss","Test Accuracy"], loc="center right")
    plt.show()

def main():
    global loss_arrayx
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500                                        # Iteration count for propagating.
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1                    # Weight value for weighted sum.
    accuracy_array = []                                  # Accuracy array for appending accuracy after each operations.
    loss_array = []                                      # Loss array for appending loss after each operations.

    train_inputs, train_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):                       #
        inputs = train_inputs                                      #
        outputs = np.dot(train_inputs,weights)                     #
        outputs = sigmoid(outputs)                                 # Forward and backward propagating part.
        loss = train_labels - outputs                              #
        tuning = loss * sigmoid_derivative(outputs)                #
        transpose_trains = np.transpose(inputs)                    #
        weights = weights + np.dot(transpose_trains,tuning)        #
        #print(iteration)
        accuracy = run_on_test_set(test_inputs, test_labels, weights)    # Getting accuracy value before appending it.

        accuracy_array.append(accuracy)                                  # Appending part.
        loss_array.append(loss.mean())                                   #

    plot_loss_accuracy(accuracy_array, loss_array)                       # Finally, plotting loss and accuracy.

if __name__ == '__main__':
    main()
