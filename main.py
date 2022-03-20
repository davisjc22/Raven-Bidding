import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, utils, backend as K
import csv
import matplotlib.pyplot as plt
#  import shap



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
                    print(hand, " ", bid)


# 500 original hands * 16 permutations for each hand = 8000 hands
# use 87.5% of hands for training (7000) and 12.5% for testing (1000)
train_bids = bids
train_bids = [int(x) for x in train_bids]
test_hands = hands[7000:]
test_bids = train_bids[7000:]
train_hands = hands[:7000]
train_bids = train_bids[:7000]


#print(len(train_bids))
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

# define metrics
def R2(y, y_hat):
    ss_res = K.sum(K.square(y - y_hat))
    ss_tot = K.sum(K.square(y - K.mean(y)))
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )


# compile the neural network
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[R2])

# train/validation
training = model.fit(x=train_hands, y=train_bids, batch_size=32, epochs=100, shuffle=True, verbose=0,
                     validation_data=(test_hands, test_bids))

# plot
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
print("finished training")

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

model.save('trained_model/bidding_model')