import os
import numpy as np
from SelfOrgControl.NeuralCA import *
from SelfOrgControl.NCA_DQN import *
import random

def show_influence_field(nca,inputs_to_sample, perturb_range=(-1.,1.),
                         nb_rounds=10, perturb_input=None, normalize_mean=False):
    """Compute the deviation of the cells when subject to a perturbation in input.

        nca -- The TrainableNeuralCA object to use
        inputs_to_sample -- the observation to sample
        perturb_range -- the range of the random perturbation (uniform)
        nb_rounds -- The number of repatition of the pertubation process on a new
                        input

        perturb_input -- The list of the inputs cell indices to perturb

        normalize_mean -- whether to normalize the mean of the norm of the difference

    """

    perturb = np.zeros((nb_rounds*nca.batch_size,nca.grid_size,
                        nca.grid_size,nca.ca_nb_channel))

    all_grids = np.zeros((2*nb_rounds*nca.batch_size,nca.grid_size,
                        nca.grid_size,nca.ca_nb_channel))

    for rd in range(nb_rounds):


        sensor_input = random.sample(inputs_to_sample,1 )[0]

        ca_input = nca.change_to_ca_shape(sensor_input, is_input=True)
        sensor_input = np.expand_dims(sensor_input, 0)

        #first baseline computation that will be the next starting point
        _, std_grids = nca.predict(sensor_input, return_final_grids=True, sample_from_pool=False)
        std_grids = np.squeeze(std_grids)

        #No perturbation
        _, std_grids2 = nca.predict(sensor_input, return_final_grids=True, sample_from_pool=False, init_grids=std_grids)
        std_grids2 = np.squeeze(std_grids2)


        perturb_grids = std_grids.copy()

        # Perturbation of the input values
        ca_input_perturb = ca_input.copy()
        ca_input_perturb[:,perturb_input] = ca_input[:,perturb_input]*np.random.uniform(perturb_range[0], perturb_range[1],
                                                                (nca.batch_size, len(perturb_input)))

        nb_ca_steps = np.random.randint(nca.ca_steps_per_sample[0],
                                        nca.ca_steps_per_sample[1])

        grid_after_perturb = nca.run_ca(ca_input_perturb, std_grids, nb_ca_steps)
        perturb_grids = np.squeeze(grid_after_perturb)

        #computation of the difference
        perturb[rd*nca.batch_size:(rd+1)*nca.batch_size,:,:,:] = std_grids2-perturb_grids

        #store all the grids value to normalize
        all_grids[rd*nca.batch_size:(rd+1)*nca.batch_size,:,:,:] = std_grids2
        all_grids[(rd+1)*nca.batch_size:(rd+2)*nca.batch_size,:,:,:] = perturb_grids


    if normalize_mean:
        mean_perturb = np.divide( np.mean( np.linalg.norm(perturb, axis=-1), axis=0 ), np.mean( np.linalg.norm(all_grids, axis=-1), axis=0 ))
    else:
        mean_perturb = np.mean( np.linalg.norm(perturb, axis=-1), axis=0 )

    return mean_perturb



### Creating video with NCA and cartpole running in parallel

CARTPOLE_WINDOW = (400,600,3)
CURVE_W = 200
CURVE_PAD = 10

inp_new_cor = [(11, 26),(25,20),(5,20),(19,6),(19,26),(5,12),(25,12) ,(11, 6)]

INPUT_LEGEND = {(11,6):"Pole\nangular\nvelocity",
                (25,12):"Pole\nangular\nvelocity",
                (5,12):"Pole\nangle",
                (19,26):"Pole\nangle",
                (5,20):"Cart\nvelocity",
                (19,6):"Cart\nvelocity",
                (25,20):"Cart\nposition",
                (11,26):"Cart\nposition"}

OUTPUT_LEGEND = [ ("Left",(13,16)), ("Right", (17,16))]

COLORS = ["green", "orange"]
MARKERS = ["<", ">"]


HOR_AL = {(5,12):'center',
(5,20): 'center',
(11,26):'left',
(19,26):'left',
(25,20):'center',
(25,12):'center',
(19,6): 'right',
(11,6):'right',
(13,16):'center',
(17,16):'center'}

VERT_AL = {(5,12):'bottom',
(5,20): 'bottom',
(11,26):'center',
(19,26):'center',
(25,20):'top',
(25,12):'top',
(19,6): 'center',
(11,6):'center',
(13,16):'bottom',
(17,16):'top'}

PAD = 13

OFFSETS = {
(5,12):(0,PAD),
(5,20): (0,PAD),
(11,26):(PAD,0),
(19,26):(PAD,0),
(25,20):(0,-PAD),
(25,12):(0,-PAD),
(19,6): (-PAD,0),
(11,6):(-PAD,0),
(13,16): (0,PAD),
(17,16): (0,-PAD)  }



def add_io_legend(axe):
    for x,y in INPUT_LEGEND:
        n = INPUT_LEGEND[(x,y)]
        axe.plot(y,x,".", color="black", alpha=0.9)
        axe.annotate(n , xy=(y,x), xytext=OFFSETS[(x,y)],
        textcoords='offset points',
                    alpha=0.9, size=11, ha=HOR_AL[(x,y)],va=VERT_AL[(x,y)])

    for i in range(2):
        n,c = OUTPUT_LEGEND[i]
        x,y = c
        axe.plot(y,x,".", color=COLORS[i], alpha=1.)
        axe.annotate(n,xy=(y,x), xytext=OFFSETS[(x,y)],
                    textcoords='offset points',
                    alpha=1., color=COLORS[i],
                    size=13, weight='extra bold',
                    ha=HOR_AL[(x,y)],va=VERT_AL[(x,y)])


VIDEO_SPEEDS = { (0,50): 1,
                (50,75):1,
                (75,90):2,
                (90,150):5,
                (150,200):20,
                (200,220):20,
                (220,240):20,
                (240,280):20,
                (280,330):50,
                (330,4000):50,
                (4000, 40000):50}

# VIDEO_SPEEDS = { (0,50): -2,
#                 (50,75):50,
#                 (75,90):50,
#                 (90,150):50,
#                 (150,200):50,
#                 (200,220):50,
#                 (220,240):50,
#                 (240,280):50,
#                 (280,330):50,
#                 (330,4000):50,
#                 (4000, 40000):50}

IMG_FOLDER = "agent_video_images/"
if not(os.path.isdir(IMG_FOLDER)):
    os.mkdir(IMG_FOLDER)

def save_this_frame(nb):
    for b,e in VIDEO_SPEEDS:
        if b<= nb and nb <e:
            if VIDEO_SPEEDS[(b,e)] <0:
                return -VIDEO_SPEEDS[(b,e)]
            else:
                return int(nb%VIDEO_SPEEDS[(b,e)] == 0)
    return 0

FRAME_NB = 0


def visualize_agent(agent,nb_steps,
                  output_video=False,
                  video_name="NCA_video", fps=25):
    global FRAME_NB
    if output_video:
        images = [IMG_FOLDER+img for img in os.listdir(IMG_FOLDER)]
        for img in images:
            os.remove(img)
        FRAME_NB = 0

    fig, axes = plt.subplots(2,2,figsize=(15,10))

    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size])

    curves_data = {"left":[], "right": []}

    #init the plots
    init_img = np.zeros(CARTPOLE_WINDOW, dtype="uint8")
    img_cartpole = axes[0][1].imshow(init_img)
    axes[0][1].tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelcolor = "w")
    axes[0][1].set_title('Cartpole environment')


    init_img = np.zeros((32,32))
    img_info_chan = axes[0][0].matshow(init_img, vmin=-0.75, vmax = 0.75, cmap='bwr')
    axes[0][0].tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=False,
        top=False,
        labelcolor = "w")
    add_io_legend(axes[0][0])
    axes[0][0].set_title('Neural cellular automaton\nInformation channel')

    init_img = np.zeros((32,32,3))
    img_hid_chan = axes[1][0].imshow(init_img)
    axes[1][0].tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=False,
        top=False,
        labelcolor = "w")
    add_io_legend(axes[1][0])
    axes[1][0].set_title('Neural cellular automaton\nHidden channels (shown as RGB values)')

    imgs = {"cartpole":img_cartpole,
            "info_chan":img_info_chan,
            "hidden_chans":img_hid_chan}

    curve_plot_left = axes[1][1].plot([0], [0], "green", label="Expected reward if LEFT")[0]
    curve_plot_right = axes[1][1].plot([0], [0], "orange", label="Expected reward if RIGHT")[0]

    curve_plots = {"left":curve_plot_left,
                    "right": curve_plot_right}


    axes[1][1].set_xlim(xmax=CURVE_W+CURVE_PAD, xmin=0)
    axes[1][1].set_ylim(ymax=70, ymin=-100)
    axes[1][1].legend(loc="lower left")
    axes[1][1].set_title('Neural cellular automaton\nOutput values')


    axes[1][1].plot([5, 5], [-1000, 1000],"--",color= "black",
                    alpha=0., lw=1., zorder = -5)
    axes[1][1].plot([5], [0],marker=MARKERS[0],
                    color= COLORS[0], alpha=0., ms=11., zorder = 5)
    #fig.tight_layout()

    agent.env.close()
    agent.env.reset()

    #we initialize the grid of the nca
    #usualy for tseting we use a batch size of 1
    agent.model.grid_pool.reinit_and_sample()

    for k in range(nb_steps):

        ca_steps = np.random.randint(50,60)
        cartpole_visu= agent.env.render(mode='rgb_array')

        if k<5:
            dam_proba = 0.015
        elif k<40:
            dam_proba = 0.002
        else:
            dam_proba = 0.002

        pred, grids = agent.model.predict(state*agent.in_factors, no_reinit=True,nb_ca_steps=ca_steps,
                                  return_all_grids=True, conti_damage_proba=dam_proba, use_batch=False)
        grids = grids[0]
        #return grids

        update_view(cartpole_visu, grids, ca_steps,
                    agent, fig, axes, curves_data,
                    curve_plots, imgs, output_video)

        action = np.argmax(pred)
        next_state, reward, done, _ = agent.env.step(action)
        state = np.reshape(next_state, [1, agent.state_size])

    if output_video:
        video_name += get_random_id()
        save_to_video(video_name,IMG_FOLDER,fps,'png')



def update_view(cartpole_visu, grids, ca_steps,
                agent, fig, axes, curves_data,
                curve_plots, imgs,output_video):
    global FRAME_NB

    imgs["cartpole"].set_data(cartpole_visu)

    for s in range(ca_steps):
        #print("grids", grids.shape)
        #update the left and right curves
        outputs = agent.model.neuralCA.get_output_cell_states(grids[s]).numpy()

        outputs = outputs[0,:,0]
        outputs = outputs/agent.out_factor

        curves_data["left"].append(outputs[0])
        curves_data["right"].append(outputs[1])

        if len(curves_data["left"]) > CURVE_W:
            beg = len(curves_data["left"])-CURVE_W
            end = len(curves_data["left"])
            axes[1][1].set_xlim(xmin=beg, xmax=end+CURVE_PAD)

        else:
            beg=0
            end = len(curves_data["left"])

        Xs = np.arange(beg, end)

        curve_plots["left"].set_xdata(Xs)
        curve_plots["right"].set_xdata(Xs)

        curve_plots["left"].set_ydata(curves_data["left"][beg:end])
        curve_plots["right"].set_ydata(curves_data["right"][beg:end])

        #update info channel

        imgs["info_chan"].set_data(grids[s][0,:,:,0])

        #update hidden channel
        hid_rgb = get_hidden_chan_rgb(grids[s])
        imgs["hidden_chans"].set_data(hid_rgb)

        #plt.pause(0.001)
        if output_video:
            nb_img = save_this_frame(end)
            if nb_img >0:
                plt.show()
            for kk in range(nb_img):
                fig.savefig(IMG_FOLDER+"/img_"+str(FRAME_NB)+".png",format='png', dpi=150, optimize=True)
                FRAME_NB +=1
                print("\rIMG #"+str(FRAME_NB), end='')

        if not(output_video):
            plt.pause(0.01)
            plt.show()

    axes[1][1].plot([end, end], [-1000, 1000],"--",color= "black",
                    alpha=0.5, lw=1., zorder = -5)
    max_val = np.max(outputs)
    idx = np.argmax(outputs)
    idx = int(idx)
    axes[1][1].plot([end], [max_val],marker=MARKERS[idx],
                    color= COLORS[idx], alpha=0.7, ms=11., zorder = 5)



def get_hidden_chan_rgb(grids, ranges=[(-1,1), (-1,1), (-1,1)]):
    """Take girds as input, output a rgb array properly scaled to display
    the value of the 3 hidden channels"""

    chans_id = [1,4,5]
    rgb = np.zeros((32,32,3), "float32")
    for c in range(3):
        chan= grids[0,:,:, chans_id[c]]
        chan = (chan-ranges[c][0])/(ranges[c][1]-ranges[c][0])
        rgb[:,:,c] = chan
    rgb = np.clip(rgb, 0,1)
    return rgb

