import matplotlib.pyplot as plt


# Diagnostics plot
def dashboard(diags):
    """
    Make a dashboard plot showing the time series of key variables from an AO box model run.
    """
    
    # Colors
    colors3 = plt.get_cmap("RdYlBu")(np.linspace(0, 1, 3))
    circ_colors = plt.get_cmap("viridis")(np.linspace(0, 1, 3))
    circ_labels = ["up", "AMOC", "south"]
    
    colors = {3:colors3,
              4:[colors3[0], colors3[1], "k", colors3[2]]}
    boxes = {3:["tropics", "north", "south"],
             4:["tropics", "north", "deep", "south"]}
    
    diag_names = [["Ta", "To", "AOQsurf"],
                  ["Qtoa", "Qmix", "OAQsurf"],
                  ["circ"]]
    units = ["degC", "W m$^{-2}$", "Sv"]
    nt = diags["psi"].shape[1]
    time = np.linspace(0, nt/360, 10)
    #time = np.arange(diags[diag_names[0][0]].shape[-1])
    
    fig, axes = plt.subplots(3, 3, figsize=(6, 4), dpi=150, sharex=True,
                             tight_layout=True)#, sharey="row")
    for ri, dnames in enumerate(diag_names):
        for ci, dname in enumerate(dnames):
            ax = axes[ri,ci]
            plt.sca(ax)
            
            data = diags[dname]
            nbox = data.shape[0]
            
            for bi, box in enumerate(boxes[nbox]):
                plt.plot(time, data[bi],
                         color=colors[nbox][bi] if dname != "circ" else circ_colors[bi],
                         label=box if dname != "circ" else circ_labels[bi])
            plt.ylabel("{} [{}]".format(dname, units[ri]))
            
            if dname == "To":
                h2, l2 = ax.get_legend_handles_labels()
            elif dname == "circ":
                h1, l1 = ax.get_legend_handles_labels()
                
            
        if ri == 2:
            plt.xlabel("time [yr]")
            
    ax1 = axes[-1,-2]
    ax1.legend(h1, l1)
    ax1.axis("off")
    
    ax2 = axes[0,-1]
    #ax2.legend(h2, l2)
    #ax2.axis("off")
    
    ax3 = axes[-1,-1]
    ax3.legend(h2, l2)
    ax3.axis("off")
    return None
    


