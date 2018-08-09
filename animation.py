# -*- coding: utf-8 -*-
"""
@author: ZhangYuning
@email: neuromancer43@gmail.com
@institute: Chongqing University

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime

def generate_animation(data,file_name,fps=60,time=10,view='upper'):
    print(file_name)
    print('Target: {}'.format(datetime.datetime.now()))
    num_iter=data.shape[0]
    step=num_iter//(fps*time)
    edge_len=int(np.sqrt(data.shape[1]))
    fig = plt.figure()
    fig.dpi=120
    ims = []
    ax=fig.add_subplot(1,1,1)
    print('Plot frames....')
    
    count=0
    for i in range(0,num_iter,step):
        count+=1
        plane=data[i].reshape(edge_len,edge_len)
        if view=='upper':
            im = plt.imshow(plane,animated=True)
        else:
            x=np.arange(edge_len)
            y=plane[:,edge_len//2]
            plt.xlim(0,edge_len)
            plt.ylim(0,edge_len)
            im,=ax.plot(x,y)
            ax.fill_between(x,y,where=y>0,facecolor='g')
        ims.append([im])
        if count%fps==0:
            print('{}/{} seconds finished!'.format(count//fps,time))
    
    print('Generating animation....')
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    
    print('Writing animation to file....')
    
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='ZhangYuning'), bitrate=-1)
    ani.save(file_name, writer=writer)
    plt.show()
    
    print('Finished!')

#%%
if __name__=='__main__':
    print('Loading data...')
    data=np.loadtxt('./data/h_data.csv')
    print('Done')
    generate_animation(data,file_name='Sandpile_side_view_shit.mp4',view='right')
    generate_animation(data,file_name='Sandpile_top_view.mp4',view='upper')
