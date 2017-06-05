from scipy.stats import norm
import matplotlib.mlab as mlab
from plotter import *


def plot_route(h, v, T_outbound, T_inbound, start_coord=np.array([0.0, 0.0]), plot_speed=False,
               plot_heading=False, memory_estimate=None, ax=None, legend=True,
               labels=True, outbound_color='purple', inbound_color='green',
               memory_color='darkorange', quiver_color='gray', nest_label=True, feeder_label=False, title=None,
               label_font_size=11, unit_font_size=10,
               figsize=(nature_single, nature_single)):
    """Plots a route with optional colouring by speed and arrows indicating
    direction."""

    xy = np.vstack([np.array([0.0, 0.0]), np.cumsum(v, axis=0)])
    x, y = xy[:, 0]+start_coord[0], xy[:, 1]+start_coord[1]

    lw = 0.5  # Linewidth
    T = T_outbound + T_inbound

    # Generate new plot if no axes passed in.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Outbound path
    if plot_speed:
        speed = np.clip(np.linalg.norm(np.vstack([np.diff(x), np.diff(y)]), axis=0), 0, 1)
        n_min = np.argmin(speed[:T_outbound])
        n_max = np.argmax(speed[:T_outbound])

        for i in range(T_outbound-1):
            ax.plot(x[i:i+2], y[i:i+2], color=(speed[i], 0.2, 1-speed[i]), lw=lw)

        blue_line = mlines.Line2D([], [], color='blue', label='Outbound (slow)')
        red_line = mlines.Line2D([], [], color='red', label='Outbound (fast)')
        handles = [blue_line, red_line]
    else:
        line_out, = ax.plot(x[0:T_outbound+1], y[0:T_outbound+1], lw=lw,
                color=outbound_color, label='Outbound')
        handles = [line_out]

    if plot_heading:
        #interval = T/200  # Good for actual route plots thing (with headwidth 0)
        interval = 20  # Good for memory plot thing (with headwidth 4)
        ax.quiver(x[1:T_outbound:interval], y[1:T_outbound:interval],
                np.sin(h[1:T_outbound:interval]),
                np.cos(h[1:T_outbound:interval]),
                pivot='tail', width=0.003, scale=12.0, headwidth=4, color=quiver_color)
                #pivot='tail', width=0.002, scale=12.0, color=quiver_color)

    # Inbound path
    if T_inbound != 0:
        line_in, = ax.plot(x[T_outbound:T], y[T_outbound:T], color=inbound_color,
                lw=lw, label='Return')

        handles.append(line_in)

    # Memory
    if memory_estimate:
        point_estimate = ax.scatter(memory_estimate[0], memory_estimate[1],
                color=memory_color, label='Memory')
        handles.append(point_estimate)

    # Nest label
    if nest_label is not False:
        ax.text(0, 0, 'N', fontsize=12, fontweight='heavy', color='k', ha='center',
                va='center')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=unit_font_size)

    if title:
        ax.set_title(title)

    if labels:
        ax.set_xlabel('Distance (steps)', fontsize=label_font_size)
        ax.set_ylabel('Distance (steps)', fontsize=label_font_size)

    # Legend
    if legend:
        l = ax.legend(handles=handles,
                      loc='best',
                      fontsize=unit_font_size,
                      handlelength=0,
                      handletextpad=0)

        if plot_speed:
            colors = ['blue', 'red', inbound_color]
        else:
            colors = [outbound_color, inbound_color]

        if memory_estimate:
            colors.append(memory_color)
        for i, text in enumerate(l.get_texts()):
            text.set_color(colors[i])
        for handle in l.legendHandles:
            handle.set_visible(False)
        l.draw_frame(False)
    return fig, ax


def plot_arena(arena, ax=None, figsize=(12,12)):

    """ Draw an arena w/ the detection zone """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Get absolute coordinates
    x1 = arena['walls'][0,0]
    x2 = arena['walls'][0,1]
    y1 = arena['walls'][1,0]
    y2 = arena['walls'][1,1]

    # Get dimensions
    w = np.diff(arena['walls'], axis=1)[0]
    h = np.diff(arena['walls'], axis=1)[1]
    d = arena['detection']

    # Draw the arena itself
    ax.add_artist(plt.Rectangle((x1, y1), w, h, color='red', alpha=0.5, fill=False))

    # Draw the detection zone
    ax.add_artist(plt.Rectangle((x1, y2-d), w, d, color='orange', alpha=0.2)) # top wall
    ax.add_artist(plt.Rectangle((x1, y1), d, h, color='orange', alpha=0.2)) # left wall
    ax.add_artist(plt.Rectangle((x1, y1), w, d, color='orange', alpha=0.2)) # bottom wall
    ax.add_artist(plt.Rectangle((x2-d, y1), d, h, color='orange', alpha=0.2)) # right wall

    return fig, ax

def plot_obstacle(obstacle, ax=None, figsize=(12,12)):

    """ Draw an arena w/ the detection zone """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if obstacle.type is 'cylinder':
        ax.add_artist(plt.Circle(obstacle.center, obstacle.detection_box, color='orange', alpha=0.2))
        ax.add_artist(plt.Circle(obstacle.center, obstacle.radius, color='red', alpha=0.5))

    elif obstacle.type == 'wall':
        # get detection box coordinates
        A = obstacle.detection_box[0]

        # get wall coordinates
        X1 = obstacle.origin
        X2 = obstacle.end

        # compute box length and width
        l = np.sqrt((X2[0] - X1[0])**2 + (X2[1] - X1[1])**2)
        w = obstacle.detection*2

        # Draw the wall itself
        plt.plot([obstacle.origin[0], obstacle.end[0]], [obstacle.origin[1], obstacle.end[1]], color='red', alpha=0.5, linestyle='-', linewidth=0.8)

        # Draw the detection box
        ax.add_artist(plt.Rectangle(A, l, w, color='orange', alpha=0.2, angle=obstacle.tilt))

    return fig, ax


def plot_traces(log, include=['TN1', 'TN2', 'CL1', 'TB1', 'CPU4', 'INH', 'CPU1', 'motor'],
                fig=None, ax=None, colormap='viridis', title_x=-0.15,
                alpha=0.2, outbound_color='purple', return_color='g',
                label_font_size=12, unit_font_size=10, dashes=[1, 2, 1, 2],
                T_almost_home=None, t_start=0, figsize=(6, 6), single_move=False):
    """Generate big plot with all traces of model. Warning: takes long time to
    save!!"""
    T, T_outbound, T_inbound = log.T, log.T_outbound, log.T_inbound
    titles = {'TN1': 'TN1 (Speed)', 'TN2': 'TN2 (Speed)', 'TL2': 'TL2',
              'CL1': 'CL1', 'TB1': 'TB1 (Compass)', 'CPU4': 'CPU4 (Integrator)',
              'INH': 'Compound vector (inhib.)', 'CPU1': 'CPU1 (Steering)', 'motor': 'motor'}
    data = {'TN1': log.tn1, 'TN2': log.tn2, 'TL2': log.tl2, 'CL1': log.cl1,
            'TB1': log.tb1, 'CPU4': log.cpu4, 'INH': log.cpu4_inh,
            'CPU1': log.cpu1, 'motor': log.motor}

    colors = {'TL2': tl2_color, 'CL1': cl1_color}

    # Generate new plot if no axes passed in.
    if ax is None:
        fig, ax = plt.subplots(len(include), 1,
                               figsize=figsize)

    N_plots = len(include)
    for i, cell_type in enumerate(include):
        ax[i].set_title(titles[cell_type],
                        x=title_x,
                        y=0.3,
                        va='center',
                        ha='right',
                        fontsize=label_font_size,
                        fontweight='heavy')
        ax[i].set_xticklabels([])
        ax[i].tick_params(labelsize=unit_font_size)

        if cell_type in ['TN1', 'TN2']:
            filtered_l = sp.ndimage.filters.gaussian_filter1d(
                    data[cell_type][0], sigma=20)
            filtered_r = sp.ndimage.filters.gaussian_filter1d(
                    data[cell_type][1], sigma=20)
            tn_l_line, = ax[i].plot(filtered_l, color=flow_color_L, label='L');
            tn_r_line, = ax[i].plot(filtered_r, color=flow_color_R, label='R');
            handles = [tn_l_line, tn_r_line]

            ax[i].plot(data[cell_type][0].T, color=flow_color_L, alpha=0.3,
                       lw=0.5);
            ax[i].plot(data[cell_type][1].T, color=flow_color_R, alpha=0.3,
                       lw=0.5);
            ax[i].set_yticks([0.05, 0.9])
            ax[i].set_yticklabels([0, 1])

            # Make a legend but not for both
            if i % 2 == 0:
                l = ax[i].legend(handles=handles,
                                 bbox_to_anchor=(1.15, 1.2),
                                 loc='upper right',
                                 ncol=1,
                                 fontsize=unit_font_size,
                                 handlelength=0,
                                 handletextpad=0)
                colors = [flow_color_L, flow_color_R]
                for i, text in enumerate(l.get_texts()):
                    text.set_color(colors[i])
                for handle in l.legendHandles:
                    handle.set_visible(False)
                l.draw_frame(False)
        elif cell_type in ['TL2', 'CL1'] and data[cell_type].shape[0] == 1:
            ax[i].plot(data[cell_type][0], color=colors[cell_type]);
            ax[i].set_yticks([-np.pi, np.pi])
            ax[i].set_yticklabels([0, 360])
        elif cell_type in ['TL2', 'CL1', 'TB1', 'CPU4', 'INH', 'CPU1']:
            # Surface plots related to memory generation.
            p = ax[i].pcolormesh(data[cell_type], vmin=0, vmax=1,
                                 cmap=colormap, rasterized=True);
            ax[i].get_xaxis().set_tick_params(direction='out')
            if cell_type == 'TB1':
                ax[i].set_yticks([1, 7])
                ax[i].set_yticklabels([1, 8])
            else:
                ax[i].set_yticks([1, 14])
                ax[i].set_yticklabels([1, 16])

            if cell_type == 'CPU1':
                # We add alpha to the outbound part
                fig.savefig('dummy.jpg')  # This is needed to force draw plot
                p.get_facecolors().reshape(16, -1, 4)[:, :T_outbound, 3] = 0.1
                p.set_edgecolor('none')
            else:
                p.set_edgecolor('face')
        else:
            # Plots related to steering
            plot_motor_trace(ax[i], log.motor, T_outbound, T_inbound,
                             outbound_color, return_color, alpha,
                             label_font_size, unit_font_size, t_start=t_start);

    # Add label half way (ish) down plot
    ax[0].set_ylabel('Activity', fontsize=label_font_size)
    #ax[1].yaxis.set_label_coords(-0.075, 1.1)

    ax[3].set_ylabel('Cell indices', fontsize=label_font_size)
    ax[3].yaxis.set_label_coords(-0.075, 1.1)

    # Add x labels to bottom plot
    ax[N_plots-1].set_xlabel('Time (steps)', fontsize=label_font_size)
    ax[N_plots-1].get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Colorbar
    ax_cb = fig.add_axes([0.92, 0.257, 0.02, 0.410])
    m = cm.ScalarMappable(cmap=colormap)
    m.set_array(np.linspace(0, 1, 100))
    ax_cb.tick_params(labelsize=unit_font_size)
    cb = fig.colorbar(m, ax_cb)
    cb.set_ticks(np.linspace(0,1,6))
    cb.set_label('Firing rate', size=label_font_size)

    # Top spacer
    ax_space = fig.add_axes([0, 0.92, 1, 0.01])
    ax_space.axis('off')

    if single_move is False:
        # Dotted bars
        if T_almost_home is None:
            T_almost_home = T_outbound + 400  # TODO (tomish) Auto generate this
        v_indices = np.array([0, T_outbound, T_almost_home, T])
        transFigure = fig.transFigure.inverted()
        for i, v_idx in enumerate(v_indices):
            y_max = ax[0].get_ylim()[1]
            coord1 = transFigure.transform(ax[0].transData.transform([v_idx,
                                                                      y_max]))
            coord2 = transFigure.transform(ax[5].transData.transform([v_idx, -3]))
            if i == 0 or i == 3:
                lw = 1
                zorder = 0
            else:
                lw = 1
                zorder = 1
            line = mlines.Line2D((coord1[0], coord2[0]),
                                 (coord1[1]+0.06, coord2[1]),
                                 transform=fig.transFigure, lw=lw, zorder=zorder,
                                 c='w', linestyle='dashed')
            line.set_dashes(dashes)
            fig.lines.append(line)
            line = ax[5].axvline(x=v_idx, lw=lw, c='#333333', linestyle='dashed')
            line.set_dashes(dashes)

        # Labels between bars
        label_indices = (v_indices[:3] + v_indices[1:])/2
        labels = ['Random Walk', 'Memory guided', 'Search']

        for i, label_idx in enumerate(label_indices):
            y_max = ax[0].get_ylim()[1]
            ax[0].text(label_idx, y_max*1.2, labels[i], fontsize=label_font_size,
                       va='center', ha='center')
    else:
        label = single_move
        y_max = ax[0].get_ylim()[1]
        ax[0].text(T/2, y_max*1.2, label, fontsize=label_font_size,
                   va='center', ha='center')

    return fig, ax

def plot_cxr_weights(cx, label_font_size=11, unit_font_size=10,
                     colormap='viridis'):
    sources = ['TL2', 'CL1', 'TB1', 'TB1', 'TN', 'TB1', 'TB1', 'CPU4', 'CPU4',
               'CPU4', 'Pontin', 'Pontin', 'INH']
    targets = ['CL1', 'TB1', 'TB1', 'CPU4', 'CPU4', 'CPU1a', 'CPU1b', 'CPU1a',
               'CPU1b', 'Pontin', 'CPU1a', 'CPU1b', 'CPU4']
    ticklabels = {'TL2': range(1, 17),
                  'CL1': range(1, 17),
                  'TB1': range(1, 9),
                  'TN': ['L', 'R'],
                  'CPU4': range(1, 17),
                  'Pontin': range(1, 17),
                  'CPU1a': range(2, 16),
                  'CPU1b': range(8, 10),
                  'INH': range(1, 17),
                 }

    weights = [-np.eye(16), cx.W_CL1_TB1, -cx.W_TB1_TB1,
               -cx.W_TB1_CPU4, cx.W_TN_CPU4, -cx.W_TB1_CPU1a,
               -cx.W_TB1_CPU1b, cx.W_CPU4_CPU1a, cx.W_CPU4_CPU1b,
               cx.W_CPU4_pontin, -cx.W_pontin_CPU1a, -cx.W_pontin_CPU1b, cx.W_LTM_CPU4]

    fig, ax = plt.subplots(5, 3, figsize=(12, 20))

    for i in range(13):
        cax = ax[i / 3][i % 3]
        p = cax.pcolor(weights[i], cmap=colormap, vmin=-1, vmax=1)
        p.set_edgecolor('face')
        cax.set_aspect('equal')

        cax.set_xticks(np.arange(weights[i].shape[1]) + 0.5)
        cax.set_xticklabels(ticklabels[sources[i]])

        cax.set_yticks(np.arange(weights[i].shape[0]) + 0.5)
        cax.set_yticklabels(ticklabels[targets[i]])

        if i == 1:
            cax.set_title(sources[i] + ' to ' + targets[i], y=1.41)
        else:
            cax.set_title(sources[i] + ' to ' + targets[i])

        cax.set_xlabel(sources[i] + ' cell indices')
        cax.set_ylabel(targets[i] + ' cell indices')
        cax.tick_params(axis=u'both', which=u'both', length=0)

    cbax = fig.add_axes([1.02, 0.05, 0.02, 0.9])
    m = cm.ScalarMappable(cmap=colormap)
    m.set_array(np.linspace(-1, 1, 100))
    cb = fig.colorbar(m, cbax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label('Connection Strength', labelpad=-50)
    cb.ax.set_yticklabels(['-1.0 (Inhibition)', '-0.5', '0.0', '0.5',
                           '1.0 (Excitation)'])
    plt.tight_layout()
    return fig, ax


def plot_success(xvar, success, ax=None, label_font_size=11, unit_font_size=10, figsize=(6,6), color='b', mode='distances'):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    mu = np.nanmean(success, axis=0)*100
    success_max = np.where(mu==max(mu))[0][0]

    ax.plot(xvar, mu, label='Success rate', color=color)
    ax.fill_between(xvar, mu, facecolor=color, alpha=0.2)
    ax.set_ylim(0, 110)

    if mode == 'gain':
        ax.set_xlim(np.nanmin(xvar), np.nanmax(xvar))
        ax.set_xlabel('Gain',
                  fontsize=label_font_size)
    else:
        ax.set_xlim(0, np.nanmax(xvar))
        ax.set_xlabel('Length of exploratory walk',
                  fontsize=label_font_size)

    ax.set_ylabel('Success rate', fontsize=label_font_size)
    ax.set_title('Success rate of memory-driven foraging', y=1.05,
                 fontsize=label_font_size)

    ax.axvline(x=xvar[success_max], ymax=max(mu)/110, color=color, linestyle='dotted')
    #ax.axhline(y=max(mu), xmin=float(distances[success_max])/float(np.nanmax(distances)), xmax=0.8, color='black', alpha=0.5, linewidth=0.5)

    if mode == 'gain':
        ax.annotate(s='{:3.1f}%'.format(np.nanmax(mu)),
                xy=(xvar[success_max], max(mu)),
                xytext=((xvar[0]+xvar[1])/2, mu[success_max]-1),
                arrowprops=dict(color='black', arrowstyle='-', linewidth=0.5, alpha=0.5),
                fontsize=10,
                color='black'
               )

        ax.annotate(s='{:3.2f}'.format(xvar[success_max]),
                    xy=(xvar[success_max], 0),
                    xytext=((xvar[success_max]+xvar[success_max+1])/2, 5),
                    arrowprops=dict(color=color, arrowstyle='simple, head_length=.3, head_width=.3, tail_width=.01'),
                    fontsize=10,
                    color=color
                   )
    else:
        ax.annotate(s='{:3.1f}%'.format(np.nanmax(mu)),
                xy=(xvar[success_max], max(mu)),
                xytext=((xvar[-3]+xvar[-2])/2, mu[success_max]-1),
                arrowprops=dict(color='black', arrowstyle='-', linewidth=0.5, alpha=0.5),
                fontsize=10,
                color='black'
               )

        ax.annotate(s='{:3.0f}'.format(xvar[success_max]),
                    xy=(xvar[success_max], 0),
                    xytext=((xvar[success_max]+xvar[success_max+1])/2, 5),
                    arrowprops=dict(color=color, arrowstyle='simple, head_length=.3, head_width=.3, tail_width=.01'),
                    fontsize=10,
                    color=color
                   )

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x) for x in vals])

    plt.tight_layout()
    return fig, ax


def plot_angular_distance_histogram(angular_distance, ax=None, bins=36, color='b', figsize=(6,6)):

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    radii = np.histogram(analysis.angular_distance,
                         np.linspace(-np.pi - np.pi / bins,
                                     np.pi + np.pi / bins,
                                     bins + 2,
                                     endpoint=True))[0]
    radii[0] += radii[-1]
    radii = radii[:-1]
    radii = np.roll(radii, bins/2)
    radii = np.append(radii, radii[0])
    # Set all values to have at least a count of 1
    # Need this hack to get the plot fill to work reliably
    radii[radii == 0] = 1
    theta = np.linspace(0, 2 * np.pi, bins+1, endpoint=True)

    width = 0

    ax = plt.subplot(111, projection='polar')
    #bars = ax.bar(theta, radii, bottom=1.0, width=width, alpha=0)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.plot(theta, radii, color=color, alpha=0.5)

    ax.set_rgrids(np.arange(50, 200, 50), size='small', alpha=0.3)
    ax.set_title('Deviation from optimal goal heading after 20 steps', y=1.08, fontsize='large')
    ax.set_rlim(0, 120)

    if color:
        ax.fill_between(theta, 0, radii, alpha=0.2, color=color)
    else:
        ax.fill_between(theta, 0, radii, alpha=0.2)

    return fig, ax


def plot_fw_route_straightness(cum_min_dist, x_count=500, ax=None,
                            label_font_size=11, unit_font_size=10, figsize=(8,6), color='b', nolegend=False):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if len(cum_min_dist.shape) > 2:
        colors = [cm.viridis(x) for x in np.linspace(0, 1, len(cum_min_dist[:,0,0]))]

        for d in range(len(cum_min_dist[:,0,0])):

            mu = np.nanmean(cum_min_dist[d,:,:], axis=1)
            sigma = np.nanstd(cum_min_dist[d,:,:], axis=1)
            t = np.linspace(0, 2, x_count)

            ax.plot(t, mu, label='Mean path', color=colors[d])
            ax.fill_between(t, mu+sigma, mu-sigma, facecolor=colors[d], alpha=0.3)

    else:
        colors = [color]

        # TESTING remove this if necessary
        mu = np.nanmean(cum_min_dist, axis=1)
        sigma = np.nanstd(cum_min_dist, axis=1)
        t = np.linspace(0, 2, x_count)

        ax.plot(t, mu, label='Mean path', color=color)
        ax.fill_between(t, mu+sigma, mu-sigma, facecolor=color, alpha=0.3)

    ax.set_ylim(0, 1)
    ax.plot([0, 1], [1, 0], 'r', label='Best possible path')
    ax.set_xlabel('Distance travelled relative to nest',
                  fontsize=label_font_size)
    ax.set_ylabel('Distance from goal', fontsize=label_font_size)
    ax.set_title('Tortuosity of foodward route', y=1.05,
                 fontsize=label_font_size)

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    ax.tick_params(labelsize=unit_font_size)

    ax.axvline(x=1, ymin=0, ymax=mu[250], color='black', linestyle='dotted')

    ax.annotate(s='',
                xy=(1, mu[250]),
                xytext=(1, 1),
                arrowprops=dict(color=color,
                                arrowstyle='<->'))

    ax.text(1.05, mu[250]+(1-mu[250])/2, '$C$', fontsize=14, color=color,
            ha='left', va='center')

    if nolegend is False:
        l = ax.legend(loc='best', prop={'size': 12}, handlelength=0,
                      handletextpad=0)

        colors.append('red')
        for i, text in enumerate(l.get_texts()):
            text.set_color(colors[i])
            text.set_ha('right')  # ha is alias for horizontalalignment
            text.set_position((103, 0))
        for handle in l.legendHandles:
            handle.set_visible(False)
        l.draw_frame(False)

    plt.tight_layout()
    return fig, ax


def plot_feeders_cloud(G, ax=None, color='b', figsize=(10,10)):

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(G[:,0], G[:,1], marker='x', color=color, s=4*figsize[0], linewidth=0.1*figsize[0], alpha=0.3)

    ax.text(0, 0, 'N', fontsize=1.2*figsize[0], fontweight='bold', color='white')

    ax.set_xlabel('Coordinates (x)')
    ax.set_ylabel('Coordinates (y)')

    ax.set_title('Feeders cloud')
    return fig, ax

def plot_feeders_hist(G, ax=None, bins=24, color='b', figsize=(10,10)):

    nb_feeders = G.shape[0]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    distances = np.zeros(nb_feeders)

    for i in range(0, nb_feeders):
        distances[i] = np.sqrt(G[i, 0]**2 + G[i, 1]**2)

    (mu, sigma) = norm.fit(distances)

    ax.hist(distances, bins, facecolor=color, alpha=0.4, rwidth=0.8, linewidth=0)

    #y = plt.normpdf(bins, mu, sigma)
    #plt.plot(bins, y, 'k--', linewidth=1.5)

    plt.xlabel('Distance')
    plt.ylabel('Occurence')

    ax.set_title('Nest-Feeder distances distribution')

    return fig, ax

def plot_relative_feeders_hist(distances, ax=None, bins=24, color='b', figsize=(10,10)):

    nb_dist = len(distances)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.hist(distances, bins, facecolor=color, alpha=0.4, rwidth=0.8, linewidth=0)

    #y = plt.normpdf(bins, mu, sigma)
    #plt.plot(bins, y, 'k--', linewidth=1.5)

    plt.xlabel('Distance')
    plt.ylabel('Occurence')

    ax.set_title('Feeder-Feeder distances distribution')

    return fig, ax
