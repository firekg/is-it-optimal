import matplotlib.pyplot as mtp


def Plot_P(p, number_observations):
      mtp.ylabel("Probability")
      mtp.xlabel("Observations")
      x = []
      for i in range(number_observations):
            x.append(i + 1)
      mtp.plot(x, p)
