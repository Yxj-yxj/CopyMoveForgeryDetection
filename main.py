import cv2
import collections
import numpy as np
import glob

def reversed_string(a_string):
    return a_string[::-1]

images_original= []
images_test = []
neighbourhood_table_original = []
neighbourhood_table_test = []

for file in glob.glob("originals/*.jpg"):
    images_original.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

if not images_original:
    exit(0)

for file2 in glob.glob("modified/*.jpg"):
    images_test.append(cv2.imread(file2, cv2.IMREAD_GRAYSCALE))

if not images_test:
    exit(0)

counter1= 0
for ig in images_original:
    rows, cols = images_original[counter1].shape
    neighbourhood_table_original.append(np.zeros((rows, cols, 1), np.uint8))
    for r in range(rows):
        for c in range(cols):
            if (c > 0 and r > 0 and c < cols - 1 and r < rows - 1):
                x = ""
                if (images_original[counter1][r - 1, c] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_original[counter1][r - 1, c + 1] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_original[counter1][r, c + 1] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_original[counter1][r + 1, c + 1] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_original[counter1][r + 1, c] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_original[counter1][r + 1, c - 1] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_original[counter1][r, c - 1] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_original[counter1][r - 1, c - 1] > images_original[counter1][r, c]):
                    x += "1"
                else:
                    x += "0"
                x = reversed_string(x)
                x = int(x, 2)
                neighbourhood_table_original[counter1][r][c] = x
    counter1 += 1


counter3= 0
for ig2 in images_test:
    rows, cols = images_test[counter3].shape
    neighbourhood_table_test.append(np.zeros((rows, cols, 1), np.uint8))
    for r in range(rows):
        for c in range(cols):
            if (c > 0 and r > 0 and c < cols - 1 and r < rows - 1):
                x = ""
                if (images_test[counter3][r - 1, c] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_test[counter3][r - 1, c + 1] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_test[counter3][r, c + 1] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_test[counter3][r + 1, c + 1] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_test[counter3][r + 1, c] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_test[counter3][r + 1, c - 1] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_test[counter3][r, c - 1] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                if (images_test[counter3][r - 1, c - 1] > images_test[counter3][r, c]):
                    x += "1"
                else:
                    x += "0"
                x = reversed_string(x)
                x = int(x, 2)
                neighbourhood_table_test[counter3][r][c] = x
    counter3 += 1

lbp_duplicated_features_train = []
counter2=0
for lbp in neighbourhood_table_original:
    block = []
    i1 = 0
    i2 = 8
    duplicated_regions = 0
    duplicated_pixels = 0
    while i2<=512:
        j1=0
        j2=8
        while j2<=512:
            block.append(neighbourhood_table_original[counter2][i1:i2,j1:j2])
            j1+=8
            j2+=8
        i1+=8
        i2+=8
    for i in range(0,len(block)):
        print(i)
        for j in range(0,len(block)):
                if i != j:
                    if np.sum(block[i] == block[j]) == 64:
                        duplicated_pixels+=1
                    if duplicated_pixels==8:
                        duplicated_regions+=1
                        duplicated_pixels = 0
    lbp_duplicated_features_train.append(duplicated_regions)
    counter2 = counter2 + 1


lbp_duplicated_features_test = []
counter4=0

for lbp2 in neighbourhood_table_test:
    block = []
    i1 = 0
    i2 = 8
    duplicated_regions = 0
    duplicated_pixels = 0
    while i2<=512:
        j1=0
        j2=8
        while j2<=512:
            block.append(neighbourhood_table_test[counter4][i1:i2,j1:j2])
            j1+=8
            j2+=8
        i1+=8
        i2+=8
    for i in range(0,len(block)):
        print(i)
        for j in range(0,len(block)):
                if i != j:
                    if np.sum(block[i] == block[j]) == 64:
                        duplicated_pixels+=1
                    if duplicated_pixels==8:
                        duplicated_regions+=1
                        duplicated_pixels = 0
    lbp_duplicated_features_test.append(duplicated_regions)
    counter4 = counter4 + 1

class classifier():
    def __init__(self):
        self.features_train = []
        self.labels_train = []
        self.features_train_true = []
        self.labels_train_true = []
        self.features_train_false = []
        self.labels_train_false = []
        self.features_test = []
        self.labels_test = []
        self.true_counter = 0
        self.false_counter = 0


    def fit(self, features_train_arg, labels_train_arg):
        self.features_train = features_train_arg
        self.labels_train = labels_train_arg

    def append(self, features_test_arg, labels_test_arg):
        self.features_test = features_test_arg
        self.labels_test = labels_test_arg

    def train_data_false_true_maker(self):
        for i in range(0, self.true_counter):
            self.features_train_true.append(0)
            self.labels_train_true.append(0)

        for i in range(0, self.false_counter):
            self.features_train_false.append(int(0))
            self.labels_train_false.append(int(0))

    def split(self):
        j = 0
        k = 0
        for i in self.labels_train:
            if i:
                self.labels_train_true[j] = True
                self.features_train_true[j] = self.features_train[i]
                j += 1
            else:
                self.labels_train_false[k] = True
                self.features_train_false[k] = self.features_train[i]
                k += 1

    def predict(self):
        predict_values = []
        for i in range(0,len(self.features_test)):
            predict_values.append(self.hamming_dist_test(self.features_test[i]))
        return predict_values

    def accurate(self):
        good_predict = 0
        count = len(self.features_test)
        predict_value = self.predict()
        for i in range(0, count):
            if self.labels_test[i] == predict_value[i]:
                good_predict += 1
        return good_predict / count

    def hamming_dist_train(self, index):
        if index > 130:
            return True
        else:
            return False

    def hamming_dist_test(self, value):
        if abs(max(self.features_train_false) - value) > abs(min(self.features_train_false) - value):
            return True
        else:
            return False

    def calculate_train(self):
        for i in lbp_duplicated_features_train:
             x1 = self.labels_train.append(self.hamming_dist_train(i))
             if x1:
                self.true_counter = self.true_counter + 1
             else:
                self.false_counter = self.false_counter + 1

    def calculate_test(self):
        for i3 in range(0,len(self.features_test)):
            self.labels_test.append(self.hamming_dist_test(self.features_test[i3]))

    def print_results(self):
        print("\n Predict %: " + str(self.accurate()))


lbp_duplicated_labels_train = []
lbp_duplicated_labels_test = []

Model = classifier()
Model.fit(lbp_duplicated_features_train,lbp_duplicated_labels_train)
Model.calculate_train()
Model.train_data_false_true_maker()
Model.split()
Model.append(lbp_duplicated_features_test,lbp_duplicated_labels_test)
Model.calculate_test()
Model.print_results()