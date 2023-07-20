import csv

a = []
print("The given training examples are:")
with open('enjoysport.csv', 'r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)
        print(row)

print("\n The total number of training instances are: ", len(a))

num_attributes = len(a[0]) - 1
print("\n The initial hypothesis is:")
hypothesis = ['0'] * num_attributes
print(hypothesis)

for i in range(0, len(a)):
    if a[i][num_attributes] == 'yes':
        for j in range(0, num_attributes):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'

    print("\n The hypothesis for the training instance {} is:\n".format(i + 1), hypothesis)

print("\n The Maximally Specific Hypothesis for the training instance is:")
print(hypothesis)
