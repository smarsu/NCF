# Copyright (c) 2020 smarsu. All Rights Reserved.

import matplotlib.pyplot as plt

x = list(range(10))
loss = [0.572, 0.492, 0.483, 0.479, 0.479, 0.479, 0.479, 0.478, 0.478, 0.477]
right_rate = [0.741, 0.748, 0.755, 0.754, 0.755, 0.755, 0.755, 0.754, 0.757, 0.757]

# plt.plot(x, loss)
plt.plot(x, right_rate)

plt.title('right_rate')
plt.savefig('img/right_rate.jpg')
