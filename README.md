# EIP4_Session2
EIP4 Session 2 - Parsimonious 12K
##################### EPOCH LOGS #########################
Epoch 00006: LearningRateScheduler setting learning rate to 0.0012269939.
60000/60000 [==============================] - 11s 184us/step - loss: 0.1358 - acc: 0.9521 - val_loss: 0.0294 - val_acc: 0.9929
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010972933.
60000/60000 [==============================] - 11s 184us/step - loss: 0.1304 - acc: 0.9517 - val_loss: 0.0314 - val_acc: 0.9920
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009923917.
60000/60000 [==============================] - 11s 183us/step - loss: 0.1199 - acc: 0.9548 - val_loss: 0.0290 - val_acc: 0.9925
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0009057971.
60000/60000 [==============================] - 11s 184us/step - loss: 0.1166 - acc: 0.9548 - val_loss: 0.0234 - val_acc: 0.9932
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0008331019.
60000/60000 [==============================] - 11s 183us/step - loss: 0.1108 - acc: 0.9565 - val_loss: 0.0229 - val_acc: 0.9936
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007712082.
60000/60000 [==============================] - 11s 183us/step - loss: 0.1089 - acc: 0.9555 - val_loss: 0.0241 - val_acc: 0.9933
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0007178751.
60000/60000 [==============================] - 11s 184us/step - loss: 0.1055 - acc: 0.9564 - val_loss: 0.0230 - val_acc: 0.9939
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006714414.
60000/60000 [==============================] - 11s 186us/step - loss: 0.1023 - acc: 0.9557 - val_loss: 0.0238 - val_acc: 0.9936
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0006306496.
60000/60000 [==============================] - 11s 184us/step - loss: 0.1028 - acc: 0.9556 - val_loss: 0.0191 - val_acc: 0.9951
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005945303.
60000/60000 [==============================] - 11s 182us/step - loss: 0.0977 - acc: 0.9572 - val_loss: 0.0226 - val_acc: 0.9937
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005623243.
60000/60000 [==============================] - 11s 182us/step - loss: 0.0979 - acc: 0.9575 - val_loss: 0.0189 - val_acc: 0.9946
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0005334282.
60000/60000 [==============================] - 11s 183us/step - loss: 0.0978 - acc: 0.9570 - val_loss: 0.0207 - val_acc: 0.9942
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0005073567.
60000/60000 [==============================] - 11s 182us/step - loss: 0.0959 - acc: 0.9576 - val_loss: 0.0207 - val_acc: 0.9944
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004837149.
60000/60000 [==============================] - 11s 182us/step - loss: 0.0941 - acc: 0.9571 - val_loss: 0.0202 - val_acc: 0.9944
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.0004621784.
60000/60000 [==============================] - 11s 183us/step - loss: 0.0930 - acc: 0.9574 - val_loss: 0.0215 - val_acc: 0.9944
############################################# END OF EPOCH LOGS #########################################

###################### MODEL EVALUATION SCORE #################################
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

[0.021548032833146862, 0.9944]
###################### END OF MODEL EVALUATION SCORE #################################

###################### STRATEGY USED #################################
1. Start with only Conv2D with 10 features
2. Every Conv2D followed by batch norm and dropout(0.1) as per strategy in notebook 8
3. Remove large Convolution block at the end and use:
     1. 2 GAP Layers
	 2. One 1x1x10 
4. Change the LR scheduler and tried with following:
	a. new_lr = round(0.003 * 1/(1 + 0.319 * epoch), 10)
	b. new_lr = round(0.003 * 1/(1 + 0.289 * epoch), 10)  --------> This was finally used as it gave the maximum accuracy (99.51 on 14th Epoch)
	c. new_lr = round(0.003 * 1/(1 + 0.300 * epoch), 10)
5. General tools such as visualizing LR/Loss etc.
6. Next steps to try different LR algorithm
###################### END OF STRATEGY USED #################################







