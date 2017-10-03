num_train = 400
num_val = 100

train_path = 'train.txt'
val_path = 'val.txt'

f = open(train_path, 'w')
for i in range(num_train):
    f.write('train/%d.jpg\n' % (i+1))
f.close()

f = open(val_path, 'w')
for i in range(num_val):
    f.write('val/%d.jpg\n' % (i+1))
f.close()