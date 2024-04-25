import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, -10, 30])
import pickle

pickle.dump(fig, open('hey.pickle', 'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`
figx = pickle.load(open('hey.pickle', 'rb'))

figx.show()

figx = pickle.load(open('figure1.pickle', 'rb'))

figx.show()
