from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Exp 3.1
# total_confusion = np.array([[27.,  5.,  3.], [ 2., 48., 13.], [ 1., 12., 33.]])
# Exp 3.4
# total_confusion = np.array([[16., 13.,  6.], [ 1., 51.,  9.], [ 2., 12., 32.]])
# Exp 3.5
total_confusion = np.array([[ 7., 25.,  3.], [ 5., 50.,  7.], [ 1., 17., 28.]])

# total_confusion = np.array([[ 56., 7.], [ 25., 21.]])
# total_confusion = np.array([[ 51., 12.], [ 12., 34.]])

normlized = total_confusion.astype('float') / total_confusion.sum(axis=1)[:, np.newaxis]


print('>> Confusion Matrix:')
print(normlized)

plt.rcParams.update({'font.size': 12})

disp = ConfusionMatrixDisplay(normlized, display_labels=['AD', 'HC', 'MCI']) # AD: 35 | CN: 63 | MCI: 46
disp.plot(cmap='Blues', colorbar=False)
plt.savefig('../confusion_matrix/Exp323E.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
