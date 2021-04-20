import matplotlib.pyplot as plt
import numpy as np
import time

def draw_model_graph(graph_name, num_epoch, y_list, graph_mode, save , zoom_plot):
    
    fig = plt.figure()
    sub_fig = fig.add_subplot(1,1,1)
    x_values = np.arange(0, num_epoch)
    
    for name, y_values in y_list.items():
        sub_fig.plot(x_values, y_values, '-', label=name)
    

    
    if graph_mode == "acc" and zoom_plot == False:
        sub_fig.set_ylim([0, 100 ])
    elif graph_mode == "acc" and zoom_plot == True:
        sub_fig.set_ylim([96, 100])
    elif graph_mode == "loss" and zoom_plot == True:
        sub_fig.set_ylim([1.470, 1.525])
    
    
    sub_fig.set_xlim([0, num_epoch])
    sub_fig.set_xlabel('epoch')
    sub_fig.set_ylabel(graph_mode)
  
    sub_fig.set_title(graph_name)
    sub_fig.legend()
    
    if save == True :
        fig.savefig('graph/'+ graph_name + time.strftime("_%H_%M_%S", time.gmtime(time.time())) +'.png')
    else:
        fig.show()
        plt.pause(10)
    
    
