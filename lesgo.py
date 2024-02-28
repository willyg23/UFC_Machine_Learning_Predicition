import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('ufc_data.csv')

# mango advice
# RELU and softmax activtion methods
# funcntional api is for more complex things. multiple inputs mapped to multiple outputs
# sequential api is less complex, one input to one output

# ade advice on what to add
# you could add batch normalization, max pulling, convoloution neural networks use many layers to predict output


#prints all the columns in df
# columns = df.columns
# print(columns)

#defines a logistical regression model
model = tf.keras.Sequential([
    # in input_shape, we have the number of our features
    tf.keras.layers.Dense(units=1, input_shape=(3099,), activation='sigmoid')
])

#compiles the model and specifies our metrics for the optimizer, loss, and accuracy metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# add finish, finish_details, finish round, etc. later.  columns DE in the csv file 
#X_train = df[['R_fighter', 'B_fighter', 'R_odds', 'B_odds', 'weight_class','gender','no_of_rounds', 'B_current_lose_streak','B_current_win_streak','B_avg_SIG_STR_landed','B_avg_SIG_STR_pct','B_avg_SUB_ATT','B_avg_TD_landed','B_avg_TD_pct','B_longest_win_streak', 'B_losses','B_total_rounds_fought','B_total_title_bouts','B_win_by_Decision_Majority','B_win_by_Decision_Split','B_win_by_Decision_Unanimous','B_win_by_KO/TKO','B_win_by_Submission','B_win_by_TKO_Doctor_Stoppage','B_wins','B_Stance','B_Height_cms','B_Reach_cms', 'R_current_lose_streak','R_current_win_streak','R_avg_SIG_STR_landed','R_avg_SIG_STR_pct','R_avg_SUB_ATT','R_avg_TD_landed','R_avg_TD_pct','R_longest_win_streak', 'R_losses','R_total_rounds_fought','R_total_title_bouts','R_win_by_Decision_Majority','R_win_by_Decision_Split','R_win_by_Decision_Unanimous','R_win_by_KO/TKO','R_win_by_Submission','R_win_by_TKO_Doctor_Stoppage','R_wins','R_Stance','R_Height_cms','R_Reach_cms', 'B_match_weightclass_rank','R_match_weightclass_rank']]
#y_train =df[['Winner']]


numerical_features = df[['R_odds', 'B_odds', 'no_of_rounds', 
                         'B_current_lose_streak', 'B_current_win_streak', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_longest_win_streak', 'B_losses', 'B_total_rounds_fought', 'B_total_title_bouts', 'B_Height_cms', 'B_Reach_cms',
                         'R_current_lose_streak', 'R_current_win_streak', 'R_avg_SIG_STR_landed', 'R_avg_SIG_STR_pct', 'R_avg_SUB_ATT', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_longest_win_streak', 'R_losses', 'R_total_rounds_fought', 'R_total_title_bouts',  'R_Height_cms', 'R_Reach_cms',
                         'B_Weight_lbs', 'R_Weight_lbs']]

categorical_features = ['R_fighter', 'B_fighter', 'weight_class', 'gender', 
                        'B_Stance', 'R_Stance',
                        'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission', 'B_win_by_TKO_Doctor_Stoppage',
                        'R_win_by_Decision_Majority', 'R_win_by_Decision_Split', 'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO', 'R_win_by_Submission', 'R_win_by_TKO_Doctor_Stoppage']

encoded_data = pd.DataFrame() # A placeholder for encoded data

for col in categorical_features:
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_col = encoder.fit_transform(df[[col]]).toarray()
    encoded_data = pd.concat([encoded_data, pd.DataFrame(encoded_col)], axis=1)


# Combine encoded and numerical features 
X = pd.concat([encoded_data, numerical_features], axis=1) 

y = df['Winner'].replace({'Red': 0, 'Blue': 1})

#
# Train-Test Split
# this splits the data into training and testing sets. test_size = 0.2 would be an 80/20 split of training to testing data. random_state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# this splits the data into training and testing sets. test_size = 0.2 would be an 80/20 split of training to testing data. random_state makes the split reproducible.
#X_train, X_test, y_train, y_test = train_test_split(df[['R_fighter', 'B_fighter', 'R_odds', 'B_odds', 'weight_class','gender','no_of_rounds', 'B_current_lose_streak','B_current_win_streak','B_avg_SIG_STR_landed','B_avg_SIG_STR_pct','B_avg_SUB_ATT','B_avg_TD_landed','B_avg_TD_pct','B_longest_win_streak', 'B_losses','B_total_rounds_fought','B_total_title_bouts','B_win_by_Decision_Majority','B_win_by_Decision_Split','B_win_by_Decision_Unanimous','B_win_by_KO/TKO','B_win_by_Submission','B_win_by_TKO_Doctor_Stoppage','B_wins','B_Stance','B_Height_cms','B_Reach_cms', 'R_current_lose_streak','R_current_win_streak','R_avg_SIG_STR_landed','R_avg_SIG_STR_pct','R_avg_SUB_ATT','R_avg_TD_landed','R_avg_TD_pct','R_longest_win_streak', 'R_losses','R_total_rounds_fought','R_total_title_bouts','R_win_by_Decision_Majority','R_win_by_Decision_Split','R_win_by_Decision_Unanimous','R_win_by_KO/TKO','R_win_by_Submission','R_win_by_TKO_Doctor_Stoppage','R_wins','R_Stance','R_Height_cms','R_Reach_cms', 'B_match_weightclass_rank','R_match_weightclass_rank']], df[['Winner']], test_size=0.2, random_state=42)

#'fit' method trains the model data. 
#first param is , second param is, 3rd , 4th, 
#An epoch is one complete pass through the entire dataset during the training process.

'''
Batch size == to the number of training examples utilized in one iteration. Instead of updating the model's parameters 
after every single example (which would be extremely slow), we update them after a batch of examples. The batch size determines how many examples are processed simultaneously before the model's parameters are updated.
'''

#train_test_split outputs both X and y features as dataFrames, but the model is expecting NumPy arrays
# So, we convert DataFrames to NumPy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

print(X_train.shape)  # Should output something like (Num_Samples, 51 + Num_Encoded_Features)
print(y_train.shape)  # Should output something like (Num_Samples,)
print("\n\nnumerical features print: \n")
print(numerical_features)

print("\n\n df.head()print: \n")
print(df.head())


model.fit(X_train, y_train, epochs=20, batch_size=128)


loss, accuracy = model.evaluate(X_test, y_test)
#print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


# here we could replace new data with a new data set of fights to predict
#predictions = model.predict(new_data)
