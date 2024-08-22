
from datetime import datetime
import pytz
import os
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 


def perform_feature_engineering(raw_data_df):
    # this is a dummy feature engineering function.
    # in real-case, it can be replace with more sophisticated code
    # that perform feature engineering and derive new features
    return raw_data_df

if __name__ == "__main__":
    
 base_dir = "/opt/ml/processing"
input_data_for_training = pd.read_csv(fr'{base_dir}/input/input-data-for-training.csv')
#input_data_for_training.head()

# x = input_data_for_training.age
# y = input_data_for_training.salary

# plt.scatter(x, y)
# plt.title("Age-Vs-Salary Distribution")
# plt.xlabel("Age")
# plt.ylabel("Salary")
# plt.show()

# Split the dataset for training
X_train, X_test, y_train, y_test = train_test_split(
                                    input_data_for_training.age, 
                                    input_data_for_training.salary,
                                    random_state=104,  
                                    test_size=0.3,  
                                    shuffle=True
                                    ) 

print(f'input_data_for_training.shape : {input_data_for_training.shape}')
print(f'X_train : {len(X_train)} y_train : {len(y_train)}')
print(f'X_test  : {len(X_test)} y_test  : {len(y_test)}')

input_data_for_training = perform_feature_engineering(input_data_for_training)
#input_data_for_training.head(5)

train_df = pd.DataFrame({'y_salary' : y_train, 'x_age' : X_train})
print(f'train_df.shape : {train_df.shape}')
#train_df.head(3)

validation_df = pd.DataFrame({'y_salary' : y_test, 'x_age' : X_test})
print(f'validation_df.shape : {validation_df.shape}')
#validation_df.head(3)

train_df.to_csv(fr'{base_dir}/train/train_df.csv', index=False, header=False)
validation_df.to_csv(fr'{base_dir}/validation/validation_df.csv', index=False, header=False)
