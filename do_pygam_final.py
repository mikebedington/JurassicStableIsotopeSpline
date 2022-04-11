import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f


def choose_sample(x,y,perc_miss=10):
    new_len = int(np.floor(len(x)*(1-(perc_miss/100))))

    orig_ind = np.arange(0,len(x))
    sel_ind = []

    for i in np.arange(0,new_len):
        choose = np.random.randint(0,len(orig_ind))
        sel_ind.append(orig_ind[choose])
        orig_ind = np.delete(orig_ind,choose)
    sel_ind = np.asarray(sel_ind)

    return x[sel_ind], y[sel_ind]

def train_gam(x,y,n_splines, lams = np.logspace(-3, 5, 50)):

    gam = LinearGAM(s(0, n_splines=n_splines)).fit(x, y)
    gam.gridsearch(x, y, lam=lams)

    return gam

def run_bootstrap(x_all, y_all, nsplines, n_samps, perc_miss):
    all_gams = []

    for this_samp in np.arange(0,n_samps):
        this_x, this_y = choose_sample(x_all, y_all,perc_miss=perc_miss)
        all_gams.append(train_gam(this_x, this_y,nsplines))

    XX = np.linspace(np.min(x_all), np.max(x_all), 500)
    YY = np.asarray([this_gam.predict(XX) for this_gam in all_gams])

    return np.std(YY, axis=0)

for var in ['d13c', 'd18o', 'SrCa']:
    do_scatter = False

    # Whether to do bootstrap confidence intervals and if so the parameters for sampling
    # If false then will use PyGAM confidence_interval function
    do_bootstrap = False
    nsamples = 100
    perc_miss = 10

    # Limits on the plots
    y_lims = [142, 184]
    if var == 'SrCa':
        x_lims = [0,3]
    else:
        x_lims = [-10,10]

    # No of splines for each fit; chosen apriori based on the time scale of the data
    # The regions are 1:Western sub-boreal, 2:North-west Tethys, 3:South-west Tethys, 4:Our own data (Greenland),
    # 5:Eastern sub-boreal, 6:Boreal (not presented as too little data)
    n_splines_region = {1:30, 2:25, 3:10, 4:10, 5:25, 6:10}

    # Load the data
    data = np.loadtxt('new_{}.csv'.format(var), delimiter=',', skiprows=1)
    region = data[:,0]
    x_all = data[:,2]
    y_all = data[:,1]

    # The X values for plotting
    XX = np.linspace(np.min(x_all), np.max(x_all), 500)

    # Fit the spline to the data from our own paper (region 4)
    x2 = x_all[region==4]
    y2 = y_all[region==4]
    region4_gam = LinearGAM(s(0, n_splines=n_splines_region[4])).fit(x2, y2)

    pre_lam = region4_gam.lam
    lams = np.logspace(-3, 5, 50)
    region4_gam.gridsearch(x2, y2, lam=lams)
    post_lam = region4_gam.lam

    print('Region {} pre fit {} after fit {}'.format(4, pre_lam, post_lam))

    if do_bootstrap:
        region4_std = run_bootstrap(x2, y2, n_splines_region[4], nsamples, perc_miss) 

    # Now do the plots against all the other regions
    plt.rcParams.update({'font.size': 22})

    i = 0

    if var in ['d18o', 'd13c']:
        fig, ax = plt.subplots(1,5,figsize=[45,20])

        for this_region in [1,2,3,5,6]:
            n_splines_rest = n_splines_region[this_region]
            x1 = x_all[region==this_region]
            y1 = y_all[region==this_region]

            # Only cover the period of region 5 which has a reasonable number of points
            if this_region == 5:
                exclude = x1>165
                x1 = x1[~exclude]
                y1 = y1[~exclude]

            this_region_gam = LinearGAM(s(0, n_splines=n_splines_rest)).fit(x1, y1)
            pre_lam = this_region_gam.lam

            lams = np.logspace(-3, 5, 50)
            this_region_gam.gridsearch(x1, y1, lam=lams)

            post_lam = this_region_gam.lam

            print('Region {} pre fit {} after fit {}'.format(this_region, pre_lam, post_lam))

            if do_bootstrap:
                this_region_std = run_bootstrap(x1, y1, n_splines_region[this_region], nsamples, perc_miss)

            this_ax = ax[i]
            this_ax.plot(this_region_gam.predict(XX), XX, 'r')

            if do_bootstrap:
                this_ax.plot(this_region_gam.predict(XX) - 2*this_region_std, XX, color='r', ls='--')
                this_ax.plot(this_region_gam.predict(XX) + 2*this_region_std, XX, color='r', ls='--')
            else:
                this_ax.plot(this_region_gam.confidence_intervals(XX), XX, color='r', ls='--')

            if do_scatter:
                this_ax.scatter(y1,x1,s=50,c='r',zorder=3)

            this_ax.plot(region4_gam.predict(XX), XX, 'b')

            if do_bootstrap:
                this_ax.plot(region4_gam.predict(XX) - 2*region4_std, XX, color='b', ls='--')
                this_ax.plot(region4_gam.predict(XX) + 2*region4_std, XX, color='b', ls='--')
            else:
                this_ax.plot(region4_gam.confidence_intervals(XX), XX, color='b', ls='--')

            if do_scatter:
                this_ax.scatter(y2,x2,s=50,c='b', zorder=3)

            this_ax.grid(axis = 'y')
            this_ax.set_ylim(y_lims)
            this_ax.set_xlim(x_lims)
            this_ax.invert_yaxis()
            this_ax.set_title('Region {}'.format(this_region))
            i += 1

        plt.tight_layout()

        if do_scatter:
            plt.savefig('final_fit_{}_s{}_{}_{}_{}_{}_scatter.pdf'.format(var,n_splines_region[4],n_splines_region[1],
                        n_splines_region[2], n_splines_region[3], n_splines_region[5]), dpi=180, transparent=True)
        else:
            plt.savefig('final_fit_{}_s{}_{}_{}_{}_{}.pdf'.format(var,n_splines_region[4],n_splines_region[1],
                        n_splines_region[2], n_splines_region[3], n_splines_region[5]), dpi=180, transparent=True)
        plt.close()

    else:
        fig, ax = plt.subplots(figsize=[10,20])

        ax.plot(region4_gam.predict(XX), XX, 'b')

        if do_bootstrap:
            ax.plot(region4_gam.predict(XX) - 2*region4_std, XX, color='b', ls='--')
            ax.plot(region4_gam.predict(XX) + 2*region4_std, XX, color='b', ls='--')
        else:
            ax.plot(region4_gam.confidence_intervals(XX), XX, color='b', ls='--')

        if do_scatter:
            ax.scatter(y2,x2,s=50,c='b', zorder=3)

        ax.grid(axis = 'y')
        ax.set_ylim(y_lims)
        ax.set_xlim(x_lims)
        ax.invert_yaxis()
        
        plt.tight_layout()

        if do_scatter:
            plt.savefig('final_fit_{}_s{}_{}_{}_{}_{}_scatter.pdf'.format(var,n_splines_region[4],n_splines_region[1],
                        n_splines_region[2], n_splines_region[3], n_splines_region[5]), dpi=180, transparent=True)
        else:
            plt.savefig('final_fit_{}_s{}_{}_{}_{}_{}.pdf'.format(var,n_splines_region[4],n_splines_region[1],
                        n_splines_region[2], n_splines_region[3], n_splines_region[5]), dpi=180, transparent=True)
