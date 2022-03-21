import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, utils, backend as K
import csv
import matplotlib.pyplot as plt
#  import shap


def R2(y, y_hat):
    # print("y: ", y, ", y_hat: ", y_hat)
    ss_res = K.sum(K.square(y - y_hat))
    ss_tot = K.sum(K.square(y - K.mean(y)))
    answer = ( 1 - ss_res/(ss_tot + K.epsilon()) )
    # print("R2 type: ", answer.dtype)
    return answer


def my_loss_fn(y_true, y_pred):
    # print("y_true: ", y_true, ", y_pred: ", y_pred)
    squared_difference = K.square(y_true - y_pred)
    answer = tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
    # print("loss type: ", answer.dtype)
    return answer


def my_metric_fn(y_true, y_pred):
    # print("y_true: ", y_true, ", y_pred: ", y_pred)
    squared_difference = tf.square(y_true - y_pred)
    answer = tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
    return answer


def get_bid(bid_val):
    bid_num = bid_val*50 + 70
    return bid_num


model_key = ['R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14',
             'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14',
             'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14',
             'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Rook']

# Read In Data
file = open('raven.csv')
csvreader = csv.reader(file)
next(csvreader)  # skip the header
hands = []
bids = []
for row in csvreader:
    # cards = row.split(",")
    # put into 5 arrays of zeroes (black, green, red, yellow and rook)
    black = [0 for i in range(10)]
    green = [0 for i in range(10)]
    red = [0 for i in range(10)]
    yellow = [0 for i in range(10)]
    rook = [0]
    del row[0]
    # print(*row)
    bid = row.pop()
    if bid != "":
        bid = float(bid)
        bid = (bid-70)/50  # normalize the bid between 70 and 120
    # print("bid: ", bid)
    for card in row:
        if card == 'Rook':
            rook[0] = 1
        else:
            details = card.split(" ")
            # print(details)
            if details[0] == 'Black':
                black[int(details[1])-5] = 1
            elif details[0] == 'Green':
                green[int(details[1])-5] = 1
            elif details[0] == 'Red':
                red[int(details[1])-5] = 1
            elif details[0] == 'Yellow':
                yellow[int(details[1]) - 5] = 1
    options = [black, green, red, yellow]  # array of four card colors
    for i in options:  # grab every permutation of colors
        options_1 = options.copy()
        options_1.remove(i)
        for j in options_1:
            options_2 = options_1.copy()
            options_2.remove(j)
            for k in options_2:
                options_3 = options_2.copy()
                options_3.remove(k)
                for l in options_3:
                    # print(i, " ", j, " ", k, " ", l)
                    hand = i + j + k + l + rook  # combine the 4 colors and rook into one list
                    hands.append(hand)
                    bids.append(bid)
                    # print(hand, " ", bid)


# 500 original hands * 16 permutations for each hand = 8000 hands
# use 87.5% of hands for training (7000) and 12.5% for testing (1000)
train_bids = bids
# train_bids = [int(x) for x in train_bids]
test_hands = hands[9600:]
test_bids = train_bids[9600:]
train_hands = hands[:9600]
train_bids = train_bids[:9600]
# print(len(hands))
# print(len(bids))
# print(len(test_hands))
# print(len(test_bids))


n_features = 41
model = models.Sequential(name="DeepNN", layers=[
    ### hidden layer 1
    layers.Dense(name="h1", input_dim=n_features,
                 units=int(round((n_features + 1) / 2)),
                 activation='relu'),
    layers.Dropout(name="drop1", rate=0.2),

    ### hidden layer 2
    layers.Dense(name="h2", units=int(round((n_features + 1) / 4)),
                 activation='relu'),
    layers.Dropout(name="drop2", rate=0.2),

    ### hidden layer 3
    layers.Dense(name="h3", units=int(round((n_features + 1) / 8)),
                 activation='relu'),
    layers.Dropout(name="drop3", rate=0.2),

    ### layer output
    layers.Dense(name="output", units=1, activation='sigmoid')
])
model.summary()

# compile the neural network
model.compile(optimizer='adam', loss="mean_absolute_error", metrics=[R2])

# train/validation
training = model.fit(x=train_hands, y=train_bids, batch_size=32, epochs=100, shuffle=False, verbose=0,
                     validation_data=(test_hands, test_bids))
#training = model.fit(x=train_hands, y=train_bids, batch_size=32, epochs=100, shuffle=True, verbose=0)

## plot
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

## training
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
ax11.legend()

## validation
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics:
    ax22.plot(training.history['val_' + metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
plt.show()

# model.save('trained_model/bidding_model')

num_correct = 0
# loop through test bids and manually print and calculate accuracy
for i in range(len(test_hands)):
    answer = model.predict([test_hands[i]])
    # print(answer)
    answer = round(answer[0][0], 1)
    # print("prediction: ", answer, "real: ", test_bids[i])
    hand_str = ""
    for j in range(len(test_hands[i])):
        if test_hands[i][j] is 1:
            hand_str = hand_str + " " + model_key[j]
    guess_bid = round(get_bid(answer), 0)
    correct_bid = get_bid(test_bids[i])
    print(hand_str, " guess bid: ", guess_bid, "| correct bid: ", correct_bid)
    if abs(guess_bid - correct_bid) <= 5.0:
        num_correct = num_correct + 1
accuracy = num_correct / len(test_hands)
print("Test Accuracy: ", accuracy)
    # do some math to allow a range of inaccuracy
    # compute overall accuracy
