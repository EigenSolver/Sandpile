# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_animation(data,file_name="Sandpile_gerenation.mp4",view='upper'):
    num_iter=data.shape[0]
    edge_len=int(np.sqrt(data.shape[1]))
    fig = plt.figure()
    fig.dpi=120
    ims = []
    print('Generating animation....')
    
    for i in range(num_iter):
        plane=data[i].reshape(edge_len,edge_len)
        if view=='upper':
            im = plt.imshow(plane,animated=True)
        else:
            x=np.arange(edge_len)
            plt.ylim(ymax=edge_len)
            im=plt.bar(x,plane[:,edge_len//2],animated=True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    
    print('Writing animation to file....')
    
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='ZhangYuning'), bitrate=-1)
    ani.save(file_name, writer=writer)
    plt.show()


if __name__=='__main__':
    print('Loading data...')
    data=np.loadtxt('h_data.csv')
    generate_animation(data,file_name='Sandpile_cut_view.mp4',view='right')
