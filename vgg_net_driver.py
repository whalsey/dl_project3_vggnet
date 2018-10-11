import tensorflow as tf
#g
# # train vggnet from scratch
# # this will be a best-effort, one-shot attempt
# from vggnet_frossard import vgg16
#
# print("initializing network")
# sess = tf.Session()
# net = vgg16(sess=sess, lr=1e-5, epochs=100, batch=50, decay=0.75)
#
# print("training")
# train_acc, valid_acc, lr_l = net.train(1)
#
# print("testing")
# test_acc = net.test_eval()
#
# with open("output_fromScratch.csv", 'w') as o:
#     buffer = ','.join(["epoch"] + [str(i) for i in range(20)])+'\n'
#     o.write(buffer)
#
#     buffer = ','.join(["training"] + [str(i) for i in train_acc])+'\n'
#     o.write(buffer)
#
#     buffer = ','.join(["validation"] + [str(i) for i in valid_acc])+'\n'
#     o.write(buffer)
#
#     buffer = ','.join(["learning_rate"] + [str(i) for i in lr_l]) + '\n'
#     o.write(buffer)
#
#     buffer = str(test_acc)+'\n'
#     o.write(buffer)
#     o.flush()


# train vggnet with pretrained weights
# will create different classes that have the different layers set to not/trainable
from vggnet_pretrained import vgg16_1

print("initializing network")
sess = tf.Session()
net = vgg16_1(weights="vgg16_weights.npz", sess=sess, lr=1e-5, epochs=100, batch=50, decay=0.75)

print("training")
train_acc, valid_acc, lr_l = net.train(1)

print("testing")
test_acc = net.test_eval()

with open("output_pretrained1.csv", 'w') as o:
    buffer = ','.join(["epoch"] + [str(i) for i in range(20)])+'\n'
    o.write(buffer)

    buffer = ','.join(["training"] + [str(i) for i in train_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["validation"] + [str(i) for i in valid_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["learning_rate"] + [str(i) for i in lr_l]) + '\n'
    o.write(buffer)

    buffer = str(test_acc)+'\n'
    o.write(buffer)
    o.flush()

from vggnet_pretrained import vgg16_2

print("initializing network")
sess = tf.Session()
net = vgg16_2(weights="vgg16_weights.npz", sess=sess, lr=1e-5, epochs=100, batch=50, decay=0.75)

print("training")
train_acc, valid_acc, lr_l = net.train(1)

print("testing")
test_acc = net.test_eval()

with open("output_pretrained2.csv", 'w') as o:
    buffer = ','.join(["epoch"] + [str(i) for i in range(20)])+'\n'
    o.write(buffer)

    buffer = ','.join(["training"] + [str(i) for i in train_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["validation"] + [str(i) for i in valid_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["learning_rate"] + [str(i) for i in lr_l]) + '\n'
    o.write(buffer)

    buffer = str(test_acc)+'\n'
    o.write(buffer)
    o.flush()

from vggnet_pretrained import vgg16_3

print("initializing network")
sess = tf.Session()
net = vgg16_3(weights="vgg16_weights.npz", sess=sess, lr=1e-5, epochs=100, batch=50, decay=0.75)

print("training")
train_acc, valid_acc, lr_l = net.train(1)

print("testing")
test_acc = net.test_eval()

with open("output_pretrained3.csv", 'w') as o:
    buffer = ','.join(["epoch"] + [str(i) for i in range(20)])+'\n'
    o.write(buffer)

    buffer = ','.join(["training"] + [str(i) for i in train_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["validation"] + [str(i) for i in valid_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["learning_rate"] + [str(i) for i in lr_l]) + '\n'
    o.write(buffer)

    buffer = str(test_acc)+'\n'
    o.write(buffer)
    o.flush()

from vggnet_pretrained import vgg16_4

print("initializing network")
sess = tf.Session()
net = vgg16_4(weights="vgg16_weights.npz", sess=sess, lr=1e-5, epochs=100, batch=50, decay=0.75)

print("training")
train_acc, valid_acc, lr_l = net.train(1)

print("testing")
test_acc = net.test_eval()

with open("output_pretrained4.csv", 'w') as o:
    buffer = ','.join(["epoch"] + [str(i) for i in range(20)])+'\n'
    o.write(buffer)

    buffer = ','.join(["training"] + [str(i) for i in train_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["validation"] + [str(i) for i in valid_acc])+'\n'
    o.write(buffer)

    buffer = ','.join(["learning_rate"] + [str(i) for i in lr_l]) + '\n'
    o.write(buffer)

    buffer = str(test_acc)+'\n'
    o.write(buffer)
    o.flush()