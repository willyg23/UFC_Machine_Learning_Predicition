import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ufc_data.csv')

# mango advice
# RELU and softmax activtion methods
# funcntional api is for more complex things. multiple inputs mapped to multiple outputs
# sequential api is less complex, one input to one output
'''
gemini note:
Your sequential model is appropriate for this scenario. Functional API is better for more complex networks with multiple inputs or outputs.
'''

'''
gemini note:
Convolutional Neural Networks (CNNs) can be useful for image or time-series data but might be less relevant here.
Batch normalization and max-pooling could be considered for larger, deeper networks, potentially improving performance.
'''

'''
gemini note softmax activation: softmax activation is used for multi-class classification (more than two outcome categories). 
Since we're predicting Winner (two classes), stick with sigmoid activation in the output layer.
Unless we do more than two classes later, then give softmax activation a try!
'''

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

'''
Numerical Features: These are already in a numerical format suitable for direct use in machine learning models: 
'''
numerical_features = df[['R_odds', 'B_odds', 'no_of_rounds', 
                         'B_current_lose_streak', 'B_current_win_streak', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_longest_win_streak', 'B_losses', 'B_total_rounds_fought', 'B_total_title_bouts', 'B_Height_cms', 'B_Reach_cms',
                         'R_current_lose_streak', 'R_current_win_streak', 'R_avg_SIG_STR_landed', 'R_avg_SIG_STR_pct', 'R_avg_SUB_ATT', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_longest_win_streak', 'R_losses', 'R_total_rounds_fought', 'R_total_title_bouts',  'R_Height_cms', 'R_Reach_cms',
                         'B_Weight_lbs', 'R_Weight_lbs']]

'''
Categorical Features: These features contain categories (text or labels)
and need to be converted to a numerical representation before being used by the model
'''
categorical_features = ['R_fighter', 'B_fighter', 'weight_class', 'gender', 
                        'B_Stance', 'R_Stance',
                        'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission', 'B_win_by_TKO_Doctor_Stoppage',
                        'R_win_by_Decision_Majority', 'R_win_by_Decision_Split', 'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO', 'R_win_by_Submission', 'R_win_by_TKO_Doctor_Stoppage']

encoded_data = pd.DataFrame() # A placeholder for encoded data
'''
This code performs one-hot encoding on categorical features. TensorFlow doesn't take in strings 
(i.e. "orthoddox" for a fighters stance), so we one-hot encodeeach string to a unique
'''

for col in categorical_features:  
    # Create an encoder to handle unknown categories:
    encoder = OneHotEncoder(handle_unknown='ignore')
    # Fit the encoder to the current column and transform it into numerical representations:
    encoded_col = encoder.fit_transform(df[[col]]).toarray()
    # Convert the encoded column into a DataFrame and append it to the encoded features:
    encoded_data = pd.concat([encoded_data, pd.DataFrame(encoded_col)], axis=1) 


# Combine encoded and numerical features 
X = pd.concat([encoded_data, numerical_features], axis=1) 

y = df['Winner'].replace({'Red': 0, 'Blue': 1})



#----- Neural Network Start -----#

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#attempt 1
#Validation Loss: 0.6706537008285522
#Validation Accuracy: 0.6112244725227356

# # setting up neural network 
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compiling the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Training the model
# history = model.fit(X_train, y_train,
#                     validation_data=(X_val, y_val),
#                     epochs=50,  # can adjust this based on how quickly the model converges
#                     batch_size=32)


#attempt 2
#Validation Loss: 0.6718082427978516
#Validation Accuracy: 0.6112244725227356
'''
changes: 
more neurons per layer (128 instead of 64)
additional dropout layer to combat potential overfitting 
adjusted the learning rate in the Adam optimizer
'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust learning rate
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32)


# Evaluating the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

#----- Neural Network End -----#



# #------Logistical Regression Start ------#
# #
# # Train-Test Split
# # this splits the data into training and testing sets. test_size = 0.2 would be an 80/20 split of training to testing data. random_state makes the split reproducible.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # this splits the data into training and testing sets. test_size = 0.2 would be an 80/20 split of training to testing data. random_state makes the split reproducible.
# #X_train, X_test, y_train, y_test = train_test_split(df[['R_fighter', 'B_fighter', 'R_odds', 'B_odds', 'weight_class','gender','no_of_rounds', 'B_current_lose_streak','B_current_win_streak','B_avg_SIG_STR_landed','B_avg_SIG_STR_pct','B_avg_SUB_ATT','B_avg_TD_landed','B_avg_TD_pct','B_longest_win_streak', 'B_losses','B_total_rounds_fought','B_total_title_bouts','B_win_by_Decision_Majority','B_win_by_Decision_Split','B_win_by_Decision_Unanimous','B_win_by_KO/TKO','B_win_by_Submission','B_win_by_TKO_Doctor_Stoppage','B_wins','B_Stance','B_Height_cms','B_Reach_cms', 'R_current_lose_streak','R_current_win_streak','R_avg_SIG_STR_landed','R_avg_SIG_STR_pct','R_avg_SUB_ATT','R_avg_TD_landed','R_avg_TD_pct','R_longest_win_streak', 'R_losses','R_total_rounds_fought','R_total_title_bouts','R_win_by_Decision_Majority','R_win_by_Decision_Split','R_win_by_Decision_Unanimous','R_win_by_KO/TKO','R_win_by_Submission','R_win_by_TKO_Doctor_Stoppage','R_wins','R_Stance','R_Height_cms','R_Reach_cms', 'B_match_weightclass_rank','R_match_weightclass_rank']], df[['Winner']], test_size=0.2, random_state=42)

# #'fit' method trains the model data. 
# #first param is , second param is, 3rd , 4th, 
# #An epoch is one complete pass through the entire dataset during the training process.

# '''
# Batch size == to the number of training examples utilized in one iteration. Instead of updating the model's parameters 
# after every single example (which would be extremely slow), we update them after a batch of examples. The batch size determines how many examples are processed simultaneously before the model's parameters are updated.
# '''






# #train_test_split outputs both X and y features as dataFrames, but the model is expecting NumPy arrays
# # So, we convert DataFrames to NumPy arrays
# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

# print(X_train.shape)  # Should output something like (Num_Samples, 51 + Num_Encoded_Features)
# print(y_train.shape)  # Should output something like (Num_Samples,)
# print("\n\nnumerical features print: \n")
# print(numerical_features)

# print("\n\n df.head()print: \n")
# print(df.head())


# model.fit(X_train, y_train, epochs=20, batch_size=128)


# loss, accuracy = model.evaluate(X_test, y_test)
# #print('Test Loss:', loss)
# print('Test Accuracy:', accuracy)

# #------Logistical Regression End ------#

# # ----- heatMap code start -----
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # Ensure 'Winner' is encoded as numbers
# df['Winner'] = df['Winner'].replace({'Red': 0, 'Blue': 1})

# selected_features = df[['Winner', 'R_odds', 'B_odds', 
#                         'R_current_win_streak', 'B_current_win_streak', 
#                         'R_current_lose_streak', 'B_current_lose_streak', 
#                         'R_Height_cms', 'B_Height_cms', 
#                         'R_Reach_cms', 'B_Reach_cms', 
#                         'R_total_rounds_fought', 'B_total_rounds_fought']]

# # Calculate the correlation matrix
# corr = selected_features.corr()

# # Generate a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation of Features with Fight Outcome')
# plt.show()
# # ----- heatMap code end -----

'''
heatMap notes:
A value close to 1 indicates a strong positive correlation... as one feature increases, the other feature tends to increase as well.
A value close to -1 indicates a strong negative correlation... meaning that as one feature increases, the other tends to decrease.
A value around 0 suggests little to no linear relationship between the features.

intersection of Winner and R_odds is 0.33:
As R_odds increase (becomes less favored to win), there's a tendency for the blue corner fighter to win more often. However, the relationship isn't very strong (since 0.33 is moderate),
Conversely, if the R_odds were lower (become more favored to win), this correlation suggests there would be a lesser tendency for the blue corner fighter to win, aligning with the expectation that lower odds indicate a higher likelihood of winning for the fighter in question.

intersection of Winner and B_odds is -0.34:
As B_odds increase (becomes less favored to win), the likelihood of the blue fighter winning decreases. This is because the odds becoming more positive (or less negative) for the blue fighter suggest they are less favored to win, which aligns with the negative correlation.
Conversely, if the B_odds are lower (become more favored to win), there is a higher likelihood of the blue fighter winning. This is indicated by the fight outcome 'Winner' being closer to 1 (blue winning).

the intersection of R_current_win_streak and B_current_win_streak == 0.36, what does that mean?
Moderate Relationship: The positive value suggests that as the win streak of the red corner fighter increases, the win streak of the blue corner fighter tends to increase as well.
Implications for Matchmaking: This correlation could suggest that fighters are often matched with opponents of similar recent success. For example, fighters on longer win streaks might be paired against each other, possibly to ensure competitive bouts.

'''




# here we could replace new data with a new data set of fights to predict
#predictions = model.predict(new_data)
