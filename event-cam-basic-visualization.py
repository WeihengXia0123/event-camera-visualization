### 
  # Exercise 01
  # This file is Weiheng Xia's homework
  # for Event Robot Vision course
  
###

# %% Import lib
import matplotlib.pyplot as plt
import numpy as np

# %% Read .txt file
data = np.loadtxt('your/file/path/xxx.txt', dtype = float)

t = data[:,0]
x = data[:,1]
y = data[:,2]
spike = data[:,3]

# Remapping of spike to [-1,+1]
for count_1 in range(spike.size):
    if(spike[count_1] == 0):
        spike[count_1] = -1
    else:
        spike[count_1] = 1

# %% Plot 3D scatter diagram
# for the first 6000 data
# Matlab Documentation: https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html
from mpl_toolkits.mplot3d import Axes3D  #Used in projection='3d'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

t_pos = []
x_pos = []
y_pos = []
t_neg = []
x_neg = []
y_neg = []
for i in range(5000):
  if(spike[i] == -1):
    t_pos.append(t[i])
    x_pos.append(x[i])
    y_pos.append(y[i])
  elif(spike[i] == 1):
    t_neg.append(t[i])
    x_neg.append(x[i])
    y_neg.append(y[i])

ax.scatter(t_pos[:5000], x_pos[:5000], y_pos[:5000], c='r',s=6) # s = size
ax.scatter(t_neg[:5000], x_neg[:5000], y_neg[:5000], c='b',s=6) 

ax.set_xlabel('time [s]')
ax.set_ylabel('X [pix]')
ax.set_zlabel('Y [pix]')
plt.gca().invert_zaxis()

plt.show()



# %% Plot 2D-Grid Image-like representation

# Step1: Sum up the polarities of each pixelpositive
# Using dictionary data structure to store sum of spikes
D = dict()
for xx,yy,ss in zip(x[:5000],y[:5000],spike[:5000]):
    D[(xx,yy)] = D.get((xx,yy),0) + ss
    # Tip 1: Dictionary
    # D.get(key, default)
    # default is 0 here, because the first time search, D doesn't have (x,y)
    # so it would return this default value


# %% Step2: Plot 2D grid
# Create an empty 2D array using np.zeros((200,250))
img = np.zeros((180,250))

for (x_temp,y_temp), s in D.iteritems():
    img[int(y_temp)][int(x_temp)] = s

fig1, ax1 = plt.subplots()
image = ax1.imshow(img,cmap='gist_gray')
fig1.colorbar(image, ax=ax1)
plt.title("Histograms of events (Event count)")
plt.show()

# %% Step3: num of pos and neg spikes 
D_pos_num = dict()
D_neg_num = dict()
num_pos = 0
num_neg = 0
# Store the number of +1 or -1 spikes into Dictionary
for x_1,y_1,s_1 in zip(x[:5000],y[:5000],spike[:5000]):
    if(s_1 == -1):
        D_neg_num[(x_1,y_1)] = D_neg_num.get((x_1,y_1),0) + 1
    elif(s_1 == 1):
        D_pos_num[(x_1,y_1)] = D_pos_num.get((x_1,y_1),0) + 1
# Iterate through the dictionary, to find max and min
for (x_1,y_1), s_1 in D_pos_num.iteritems():
    if(s_1 >= num_pos):
        num_pos = s_1
for (x_1,y_1), s_1 in D_neg_num.iteritems():
    if(s_1 >= num_pos):
        num_neg = s_1

print "The three colors represent the number of positive/negative spikes at that pixel"
print "the maximum number of positive events at any pixel: ", num_pos
print "the minimum number of positive events at any pixel: ", num_neg



# %% Plot Pseudo Color Blue-White-Red
fig2, ax2 = plt.subplots()
image = ax2.imshow(img,cmap='seismic')
fig2.colorbar(image, ax=ax2)
plt.title("pseudo color")
plt.show()



# %% Plot last event polarity at each pixel

# Step1: Store the number of +1 or -1 spikes into Dictionary
D_last = dict()
for x_1,y_1,s_1 in zip(x[:5000],y[:5000],spike[:5000]):
    D_last[(x_1,y_1)] = s_1

# Step2: Plot 2D grid
# Create an empty 2D array using np.zeros((200,250))
img_1 = np.zeros((180,250))

for (x_temp,y_temp), s in D_last.iteritems():
    img_1[int(y_temp)][int(x_temp)] = s

fig1, ax1 = plt.subplots()
image_1 = ax1.imshow(img_1,cmap='gist_gray')
fig1.colorbar(image_1, ax=ax1)
plt.title("last event polarity at each pixel")
plt.show()


# %% Plot histogram of positive/negative events

# Step1: plot postive
img_1 = np.zeros((180,250))
for (x_temp,y_temp), s in D_pos_num.iteritems():
    img_1[int(y_temp)][int(x_temp)] = s

fig1, ax1 = plt.subplots()
image_1 = ax1.imshow(img_1)
fig1.colorbar(image_1, ax=ax1)
plt.title("Histogram of positive events")
plt.show()

# Step2: plot negative
img_1 = np.zeros((180,250))
for (x_temp,y_temp), s in D_neg_num.iteritems():
    img_1[int(y_temp)][int(x_temp)] = s

fig1, ax1 = plt.subplots()
image_1 = ax1.imshow(img_1)
fig1.colorbar(image_1, ax=ax1)
plt.title("Histogram of negative events")
plt.show()


# %% Plot Images of exp decay timestamps of both polarities
# Step1: store data into a dictionary called, D_timeStamp[(x,y):t]
#        choose the latest event at each pixel
D_timeStamp = dict()
for x_timeStamp,y_timeStamp,timeStamp in zip(x[:50000],y[:50000],t[:50000]):
    if( timeStamp > D_timeStamp.get((x_timeStamp,y_timeStamp),0) ):
        D_timeStamp[(x_timeStamp,y_timeStamp)] = timeStamp


# Step2: exponential equation
img_timeStamp = np.zeros((180,250))
for (x_timeStamp,y_timeStamp), timeStamp in D_timeStamp.items():
    img_timeStamp[int(y_timeStamp)][int(x_timeStamp)] = \
        np.exp(-np.abs( t[49999]-D_timeStamp.get((x_timeStamp,y_timeStamp),0) )/0.030)

# Step3: plot the image
fig, ax = plt.subplots()
image = ax.imshow(img_timeStamp)
fig.colorbar(image, ax=ax)
plt.title("Time surface(exp decay). Both polarities")
plt.show()  

# %% Plot Time surface of exp decay timestamps of positive events
# Step1: store data in dict
D_timeStamp_pos = dict()
for x_timeStamp,y_timeStamp,timeStamp, spike_timeStamp in zip(x[:50000],y[:50000],t[:50000], spike[:50000]):
    if( spike_timeStamp == 1):
        if( timeStamp > D_timeStamp_pos.get((x_timeStamp,y_timeStamp),0) ):
            D_timeStamp_pos[(x_timeStamp,y_timeStamp)] = timeStamp

# Step2: exponential equation
img_timeStamp = np.zeros((180,250))
for (x_timeStamp,y_timeStamp), timeStamp in D_timeStamp_pos.items():
    img_timeStamp[int(y_timeStamp)][int(x_timeStamp)] = \
        np.exp(-np.abs( t[49999]-D_timeStamp.get((x_timeStamp,y_timeStamp),0) )/0.030)   

# Step3: plot the image
fig, ax = plt.subplots()
image = ax.imshow(img_timeStamp)
fig.colorbar(image, ax=ax)
plt.title("Time surface(exp decay) of positive events")
plt.show()  

# %% Plot Time surface of exp decay timestamps of negative events
# Step1: store data in dict
D_timeStamp_neg = dict()
for x_timeStamp,y_timeStamp,timeStamp, spike_timeStamp in zip(x[:50000],y[:50000],t[:50000], spike[:50000]):
    if( spike_timeStamp == -1):
        if( timeStamp > D_timeStamp_neg.get((x_timeStamp,y_timeStamp),0) ):
            D_timeStamp_neg[(x_timeStamp,y_timeStamp)] = timeStamp

# Step2: exponential equation
img_timeStamp = np.zeros((180,250))
for (x_timeStamp,y_timeStamp), timeStamp in D_timeStamp_neg.items():
    img_timeStamp[int(y_timeStamp)][int(x_timeStamp)] = \
        np.exp(-np.abs( t[49999]-D_timeStamp.get((x_timeStamp,y_timeStamp),0) )/0.030)   

# Step3: plot the image
fig, ax = plt.subplots()
image = ax.imshow(img_timeStamp)
fig.colorbar(image, ax=ax)
plt.title("Time surface(exp decay) of negative events")
plt.show()  

# %% Plot Images of average timestamps of both polarities in the last 0.030 s for each pixel event

# Step1: store data into a dictionary [(x,y):t]
#        choose the recent 0.030 second event at each pixel
D_num_30 = dict()
# Count the number of last 30 ms' spikes
for xx,yy,tt in zip(x[:50000],y[:50000],t[:50000]):
    if(D_timeStamp[(xx,yy)] - tt <= 0.030 ): 
        D_num_30[(xx,yy)] = D_num_30.get((xx,yy),0) + 1
# Sum of the last 30ms time stamps
D_sum_timeStamp = dict()
for xx,yy,tt,ss in zip(x[:50000],y[:50000],t[:50000],spike[:50000]):
    if(D_timeStamp[(xx,yy)] - tt <= 0.030 ): 
        D_sum_timeStamp[(xx,yy)] = D_sum_timeStamp.get((xx,yy),0) + tt
# Average the last 30ms time stamps
D_avg_timeStamp = dict()
for (xx,yy),tt in D_sum_timeStamp.items():
    D_avg_timeStamp[(xx,yy)] = tt/D_num_30[(xx,yy)]

# Step2: exponential equation
img_timeStamp = np.zeros((180,250))
for (x_timeStamp,y_timeStamp), timeStamp in D_avg_timeStamp.items():
    img_timeStamp[int(y_timeStamp)][int(x_timeStamp)] = D_avg_timeStamp.get((x_timeStamp,y_timeStamp),0)

# Step3: plot the image
fig, ax = plt.subplots()
image = ax.imshow(img_timeStamp)
fig.colorbar(image, ax=ax)
plt.title("Average timestamp regardless of polarities")
plt.show()

# %% Plot Images of average timestamps of positive polarities in the last 0.030 s for each pixel event

# Step1: store data into a dictionary [(x,y):t]
#        choose the recent 0.030 second event at each pixel
D_num_30 = dict()
# Count the number of last 30 ms' spikes
for xx,yy,tt in zip(x[:50000],y[:50000],t[:50000]):
    if(D_timeStamp[(xx,yy)] - tt <= 0.030 ): 
        D_num_30[(xx,yy)] = D_num_30.get((xx,yy),0) + 1
# Sum of the last 30ms time stamps
D_sum_pos_timeStamp = dict()
for xx,yy,tt,ss in zip(x[:50000],y[:50000],t[:50000],spike[:50000]):
    if(ss == 1):
        if(D_timeStamp[(xx,yy)] - tt <= 0.030 ): 
            D_sum_pos_timeStamp[(xx,yy)] = D_sum_pos_timeStamp.get((xx,yy),0) + tt


# Average the last 30ms time stamps
D_avg_pos_timeStamp = dict()
for (xx,yy),tt in D_sum_pos_timeStamp.items():
    D_avg_pos_timeStamp[(xx,yy)] = tt/D_num_30[(xx,yy)]

# Step2: exponential equation
img_timeStamp = np.zeros((180,250))
for (x_timeStamp,y_timeStamp), timeStamp in D_avg_pos_timeStamp.items():
    img_timeStamp[int(y_timeStamp)][int(x_timeStamp)] = D_avg_pos_timeStamp.get((x_timeStamp,y_timeStamp),0)

# Step3: plot the image
fig, ax = plt.subplots()
image = ax.imshow(img_timeStamp)
fig.colorbar(image, ax=ax)
plt.title("Average timestamp of positive polarities")
plt.show()


# %% Plot Images of average timestamps of negative polarities in the last 0.030 s for each pixel event

# Step1: store data into a dictionary [(x,y):t]
#        choose the recent 0.030 second event at each pixel
D_num_30 = dict()
# Count the number of last 30 ms' spikes
for xx,yy,tt in zip(x[:50000],y[:50000],t[:50000]):
    if(D_timeStamp[(xx,yy)] - tt <= 0.030 ): 
        D_num_30[(xx,yy)] = D_num_30.get((xx,yy),0) + 1
# Sum of the last 30ms time stamps
D_sum_neg_timeStamp = dict()
for xx,yy,tt,ss in zip(x[:50000],y[:50000],t[:50000],spike[:50000]):
    if(ss == -1):
        if(D_timeStamp[(xx,yy)] - tt <= 0.030 ): 
            D_sum_neg_timeStamp[(xx,yy)] = D_sum_neg_timeStamp.get((xx,yy),0) + tt


# Average the last 30ms time stamps
D_avg_neg_timeStamp = dict()
for (xx,yy),tt in D_sum_neg_timeStamp.items():
    D_avg_neg_timeStamp[(xx,yy)] = tt/D_num_30[(xx,yy)]

# Step2: exponential equation
img_timeStamp = np.zeros((180,250))
for (x_timeStamp,y_timeStamp), timeStamp in D_avg_neg_timeStamp.items():
    img_timeStamp[int(y_timeStamp)][int(x_timeStamp)] = D_avg_neg_timeStamp.get((x_timeStamp,y_timeStamp),0)

# Step3: plot the image
fig, ax = plt.subplots()
image = ax.imshow(img_timeStamp)
fig.colorbar(image, ax=ax)
plt.title("Average timestamp of negative polarities")
plt.show()

# %% Plot 3D scatter of only 2000 evetns
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(111, projection='3d')

t_pos = []
x_pos = []
y_pos = []
t_neg = []
x_neg = []
y_neg = []
for i in range(2000):
  if(spike[i] == -1):
    t_pos.append(t[i])
    x_pos.append(x[i])
    y_pos.append(y[i])
  elif(spike[i] == 1):
    t_neg.append(t[i])
    x_neg.append(x[i])
    y_neg.append(y[i])

ax.scatter(t_pos[:2000], x_pos[:2000], y_pos[:2000], c='r',s=6) # s = size
ax.scatter(t_neg[:2000], x_neg[:2000], y_neg[:2000], c='b',s=6) 

ax.set_xlabel('time [s]')
ax.set_ylabel('X [pix]')
ax.set_zlabel('Y [pix]')
plt.gca().invert_zaxis()

ax.view_init(azim=-50, elev=30)
plt.show()
'''
# Uncomment below code to produce 50 images and
# Automatically produce a video with ffmpeg
filename_prefix = './Result/video_img/'how to change the value in a 3D voxel
filename_suffix = '.png'
num_of_img = 50
for idx in range(0,num_of_img+1):
    ax.view_init(azim=-50+idx*(50.0/num_of_img), elev=30-idx*(30.0/num_of_img))
    filename = filename_prefix + ("%02d" % idx) + filename_suffix
    plt.savefig(filename)
    
import os
os.system('cat ./Result/video_img/* | ffmpeg -framerate 10 -i - output.avi')
'''

# %% Plot Voxel Grid
# compute histogram using the numpy function
t_bin = 5
x_bin = 250
y_bin = 180
hist, edges = np.histogramdd((t[:2000],x[:2000],y[:2000]),bins=(t_bin,x_bin,y_bin))


# plot histogram
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(hist)
ax.set(xlabel='t', ylabel='x', zlabel='y')
plt.gca().invert_zaxis()
plt.show()


# %% Count the min and max number of events in a voxel
'''
# Pseudo Code: 
1. create a dictionary of all voxels, key:(x_edge,y_edge,t_edge), value: number of voxels
2. iterate through x_edge,y_edge,t_edge and compare where the new xx,yy,tt fit in
3. iterate through dictionary and find the max and min of value
'''   
# Select data in to dictionary
D_voxel = dict()
for xx,yy,tt,ss in zip(x[:2000],y[:2000],t[:2000],spike[:2000]):
    for t_idx in range(t_bin):
        if(tt < edges[0][t_idx] or tt > edges[0][t_idx+1]):
            continue
        else:
            t_edge = edges[0][t_idx]
            for x_idx in range(x_bin):
                if(xx < edges[1][x_idx] or xx > edges[1][x_idx+1]):
                    continue
                else:
                    x_edge = edges[1][x_idx]
                    for y_idx in range(y_bin):
                        if(yy < edges[2][y_idx] or yy > edges[2][y_idx+1]):
                            continue
                        else:
                            y_edge = edges[2][y_idx]
                            D_voxel[(x_edge,y_edge,t_edge)] = D_voxel.get((x_edge,y_edge,t_edge),0) + 1 
# Max and Min number
voxel_max = 0
voxel_min = 0
for count_ in D_voxel.values():
    if(voxel_max < count_):
        voxel_max = count_
    if(voxel_min > count_):
        voxel_min = count_
print "The max number of events in a voxel: ", voxel_max
print "The min number of events in a voxel: ", voxel_min
# Voxel occupied rate
print "occupied voxel number: ", len(D_voxel)
print "voxel occupancy rate: ", round(len(D_voxel)/ float(t_bin*x_bin*y_bin), 8)  * 100, "%"

# %% Plot linear voting scheme 3D voxel grid
hist_linear = np.zeros(shape=(5,250,180))
left_vote = 0
right_vote = 0
voxel_len = edges[0][1] - edges[0][0]

t_1_mid = edges[0][0]+voxel_len/2.0
t_5_mid = edges[0][4]+voxel_len/2.0

for x_,y_,t_,pol in zip(x[:2000],y[:2000],t[:2000],spike[:2000]):
# Step1: calculate the distance from an event to nearby 2 time voxels midpoints
# Step2: distribute the vote, into the new hist 3D array "hist_linear": (15, 250, 180)
    t_edge = np.searchsorted(edges[0], t_, side='right')-1 # Tricky: if not add side='right'. The first one would be -1
    x_edge = np.searchsorted(edges[1], x_)-1
    y_edge = np.searchsorted(edges[2], y_)-1
    t_mid = edges[0][t_edge] + voxel_len/2
    if(t_edge == 5):
        t_edge = 4
    #print t_edge
    
    # calculate and store the vote into hist_linear array
    if(t_ <= t_1_mid):
        right_vote = 1 - (t_mid-t_)/voxel_len
        hist_linear[t_edge][x_edge][y_edge] += pol*right_vote
    elif(t_ >= t_5_mid):
        left_vote = 1 - (t_-t_mid)/voxel_len
        hist_linear[t_edge][x_edge][y_edge] += pol*left_vote
    else:
        if(t_ <= t_mid):
            left_vote = (t_mid-t_)/voxel_len
            right_vote = 1 - left_vote
            hist_linear[t_edge][x_edge][y_edge] += pol*right_vote
            hist_linear[t_edge-1][x_edge][y_edge] += pol*left_vote
        else:
            right_vote = (t_-t_mid)/voxel_len
            left_vote = 1 - right_vote
            hist_linear[t_edge][x_edge][y_edge] += pol*left_vote
            hist_linear[t_edge+1][x_edge][y_edge] += pol*right_vote

# Step3: remapping the "hist_linear" to the new "face_color" array: 0 ~ +1
hist_width = np.amax(hist_linear) - np.amin(hist_linear)

hist_linear_normalized = (hist_linear-np.amin(hist_linear))/hist_width

# Check the max and min of the normalized hist
print "normalized hist's max and min: ", np.amax(hist_linear_normalized), np.amin(hist_linear_normalized)

# Step4: Plot the voxel grid with color: -1 ~ +1
rc = hist_linear_normalized
gc = hist_linear_normalized
bc = hist_linear_normalized

colors = np.zeros(hist_linear_normalized.shape + (3,))
colors[:,:,:,0] = rc
colors[:,:,:,1] = gc
colors[:,:,:,2] = bc

print colors.shape
color_plain = np.zeros(hist_linear_normalized.shape + (3,))
print color_plain.shape


# plot histogram
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(hist_linear, facecolors = colors)
ax.set(xlabel='t', ylabel='x', zlabel='y')
plt.gca().invert_zaxis()
plt.show()

# max and min balance of each voxel
print "max balance: ", np.amax(hist_linear)
print "min balance: ", np.amin(hist_linear)
