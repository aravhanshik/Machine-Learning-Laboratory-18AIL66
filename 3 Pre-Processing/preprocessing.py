import pandas as pd

data = pd.read_csv('employee.csv')
print("Original Dataset:")
print(data)
print()

data.drop_duplicates(inplace=True)
print("Dataset after deleting duplicate rows:")
print(data)
print()

columns_to_delete = []
for column in data.columns:
    if data[column].nunique() == 1:
        columns_to_delete.append(column)
data.drop(columns=columns_to_delete, inplace=True)
print("Dataset after deleting columns with a single value:")
print(data)
print()

data.to_csv('preprocessed_dataset.csv', index=False)
