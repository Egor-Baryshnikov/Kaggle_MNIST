import matplotlib.pyplot as plt

def visualize_loss(losses_dict, plot=True):
    if plot:
        plt.figure(figsize=(25, 8))
        for key, loss in losses_dict.items():
            plt.plot(loss, label='{} loss: {:.4f}'.format(key, loss[-1]))
    else:
        for key, loss in losses_dict.items():
            print('{} loss: {:.4f}'.format(key, loss[-1]))