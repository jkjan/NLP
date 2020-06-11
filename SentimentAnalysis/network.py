from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, BatchNormalization,GRU
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os
sequence_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = Embedding(vocab_size, 128, input_length=max_len)(sequence_input)
print(embedded_sequences.shape)
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
    (LSTM
     (128,
      dropout=0.3,
      return_sequences=True,
      return_state=True,
      recurrent_activation='relu',
      recurrent_initializer='glorot_uniform'))(embedded_sequences)

state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
state_c = Concatenate()([forward_c, backward_c]) # 셀 상태

lstm = tf.reduce_sum(lstm, axis=1)
hidden = BatchNormalization()(lstm)
output = Dense(2, activation='softmax')(hidden)
model = Model(inputs=sequence_input, outputs=output)
Adam = optimizers.Adam(lr=0.0001, clipnorm=1.)
model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), verbose=1)