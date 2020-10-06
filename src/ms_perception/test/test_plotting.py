#!/home/msasrock/.virtualenvs/ros-melodic-venv/bin/python
import rospy
import sys, os
import numpy as np
import matplotlib.pyplot as plt

data = [
	[0.13440813, 0.23455119, 0.54589725, 0.90325129, 0.09784462, 0.60018885, 0.88336116, 0.07472809, 0.27266243, 0.34085074],
	[0.7751956, 0.41094497, 0.03918678, 0.20280729, 0.05993535, 0.77886951, 0.18973139, 0.91025519, 0.70858186, 0.23808327],
	[0.10040369, 0.12210236, 0.88172877, 0.74153805, 0.34607506, 0.34351084, 0.01183182, 0.48308039, 0.14972958, 0.83175337],
	[0.71129531, 0.51670271, 0.95762128, 0.3425177, 0.63755161, 0.12974429, 0.719073, 0.749026, 0.356224, 0.15917912],
	[0.84555405, 0.87617743, 0.71911305, 0.13477843, 0.16257171, 0.88225901, 0.88342577, 0.1770549, 0.52066517, 0.68136555],
	[0.59783822, 0.67228901, 0.38380688, 0.34037465, 0.30615267, 0.56211609, 0.95841318, 0.68871313, 0.27011088, 0.47702965]
]

input_data = np.array(data)
predictions = np.array([2.5597557e-09, 5.5880481e-01, 1.1431176e-02, 5.4749949e-03, 4.2428899e-01])
events = np.array(["unk", "stop", "move", "danger", "hit"])

fig, axes = plt.subplots(1, 2)
while True:
	if (predictions is not None) and (input_data is not None):
		shape = input_data.shape
		index = 0
		predicted_label = np.argmax(predictions)
		# Show image
		im = axes[index].imshow(input_data, cmap='gray')
		axes[index].grid(True)
		axes[index].axis('off')
		axes[index].set_title("Predicted {} ({})".format(predicted_label, predictions.shape[0]))
		# Show predictions
		index = 1
		axes[index].grid(True)
		N = predictions.shape[0]
		axes[index].set_ylabel("Probability")
		axes[index].set_xticks(np.arange(N))
		axes[index].set_xticklabels(events)
		axes[index].set_yticks(np.arange(0.0, 1.0, 0.1))
		myplot = axes[index].bar(range(N), predictions, color="#777777")
		axes[index].set_ylim([0, 1])
		myplot[predicted_label].set_color('red')
		axes[index].set_title("score: {:.3f}".format(predictions[predicted_label]))
		plt.tight_layout()
		plt.pause(0.01)
