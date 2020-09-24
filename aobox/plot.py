# plot.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize


def schematic(data, xmax=20, ymax=10, fs=22.0,
              atm_box_color='skyblue', ocn_box_color='royalblue', ax=None):
    """Make an annotated schematic."""
    # Unpack data
    Ta, To, S, D = data[:3], data[3:7], data[7:11], data[11]
    Psi, Qsurf = data[12:15], data[15:]
    
    dx = xmax*0.01
    dy = ymax*0.01

    # cmaps
    temp_cmap = plt.get_cmap('plasma')
    temp_norm = Normalize(vmin=0.0, vmax=30.0)
    salt_cmap = plt.get_cmap('viridis')
    salt_norm = Normalize(vmin=25.0, vmax=45.0)

    # Add boxes
    SL = patches.Rectangle((0, 0), xmax*0.05, ymax*0.5, lw=5,
                           edgecolor='0.5', hatch='/', facecolor='none')
    NL = patches.Rectangle((xmax*0.95, 0), xmax*0.05, ymax*0.5, lw=5,
                           edgecolor='0.5', hatch='/', facecolor='none')

    AS = patches.Rectangle((0, ymax*0.5), xmax*0.25, ymax*0.5, lw=5,
                           edgecolor=atm_box_color, facecolor='none')
    AT = patches.Rectangle((xmax*0.25, ymax*0.5), xmax*0.5, ymax*0.5, lw=5,
                           edgecolor=atm_box_color, facecolor='none')
    AN = patches.Rectangle((xmax*0.75, ymax*0.5), xmax*0.25, ymax*0.5, lw=5,
                           edgecolor=atm_box_color, facecolor='none')

    OS = patches.Rectangle((0.05*xmax, ymax*0.2), xmax*0.2, ymax*0.3, lw=5,
                           edgecolor=ocn_box_color, facecolor='none')
    yt = ymax*(0.5 - (D/500)*0.3)
    OT = patches.Rectangle((xmax*0.25, yt), xmax*0.5, ymax*0.5 - yt, lw=5,
                           edgecolor=ocn_box_color, facecolor='none')
    ON = patches.Rectangle((xmax*0.75, ymax*0.2), xmax*0.2, ymax*0.3, lw=5,
                           edgecolor=ocn_box_color, facecolor='none')
    OD = patches.Rectangle((xmax*0.05, 0.0), xmax*0.9, ymax*0.5, lw=5,
                           edgecolor=ocn_box_color, facecolor='none')

    boxes = [SL, NL, AS, AT, AN, OS, OT, ON, OD]

    for box in boxes:
        ax.add_patch(box)

    # Add labels
    # D
    ax.annotate('', (0.35*xmax, yt), (0.35*xmax, 0.5*ymax),
                arrowprops={'arrowstyle': '<->', 'lw': 4, 'color': '0.5'})
    ax.text(0.35*xmax + dx, 0.4*ymax, r'$D = {:.0f}$ m'.format(D), ha='left',
            fontsize=fs, color='0.5')

    # Circulation components
    # - Psi_south
    ax.annotate('', (0.2*xmax, 0.3*ymax), (0.2*xmax, 0.1*ymax),
                arrowprops={'arrowstyle': '->', 'lw': 0.4*Psi[2]})
    ax.annotate('', (0.3*xmax, 0.4*ymax), (0.2*xmax, 0.4*ymax),
                arrowprops={'arrowstyle': '->', 'lw': 0.4*Psi[2]})
    ax.text(0.2*xmax, 0.35*ymax,
            r'$\Psi_{south}$'+'\n'+'$= {:.1f}$ Sv'.format(Psi[2]),
            ha='center', va='center', fontsize=fs)

    # - Psi_AMOC
    ax.annotate('', (0.8*xmax, 0.3*ymax), (0.8*xmax, 0.1*ymax),
                arrowprops={'arrowstyle': '<-', 'lw': 0.4*Psi[1]})
    ax.annotate('', (0.7*xmax, 0.4*ymax), (0.8*xmax, 0.4*ymax),
                arrowprops={'arrowstyle': '<-', 'lw': 0.4*Psi[1]})
    ax.text(0.8*xmax, 0.35*ymax,
            r'$\Psi_{AMOC}$'+'\n'+'$= {:.1f}$ Sv'.format(Psi[1]),
            ha='center', va='center', fontsize=fs)

    # - Psi_up
    ax.annotate('', (0.5*xmax, 0.2*ymax), (0.5*xmax, 0.35*ymax),
                arrowprops={'arrowstyle': '<-', 'lw': 0.4*Psi[0]})
    ax.text(0.5*xmax, 0.2*ymax - dy,
            r'$\Psi_{up}$'+'\n'+'$= {:.1f}$ Sv'.format(Psi[0]),
            ha='center', va='top', fontsize=fs)

    # HF comps
    ax.annotate('', (0.15*xmax, 0.475*ymax), (0.15*xmax, 0.525*ymax),
                arrowprops={'arrowstyle': '<-' if Qsurf[2] < 0 else '->',
                            'lw': 0.4*Qsurf[2],
                            'color': 'r'})
    ax.text(0.15*xmax, 0.55*ymax + dy,
            r'$Q_{surf,S}$'+'\n'+'$= {:.1f}$ W m'.format(Qsurf[2])+r'$^{-2}$',
            ha='center', va='bottom', fontsize=fs, color='r')

    ax.annotate('', (0.5*xmax, 0.475*ymax), (0.5*xmax, 0.525*ymax),
                arrowprops={'arrowstyle': '<-' if Qsurf[0] < 0 else '->',
                            'lw': 0.4*Qsurf[0],
                            'color': 'r'})
    ax.text(0.5*xmax, 0.55*ymax + dy,
            r'$Q_{surf,T}$'+'\n'+'$= {:.1f}$ W m'.format(Qsurf[0])+r'$^{-2}$',
            ha='center', va='bottom', fontsize=fs, color='r')

    ax.annotate('', (0.85*xmax, 0.475*ymax), (0.85*xmax, 0.525*ymax),
                arrowprops={'arrowstyle': '<-' if Qsurf[1] < 0 else '->',
                            'lw': 0.4*Qsurf[1],
                            'color': 'r'})
    ax.text(0.85*xmax, 0.55*ymax + dy,
            r'$Q_{surf,N}$'+'\n'+'$= {:.1f}$ W m'.format(Qsurf[1])+r'$^{-2}$',
            ha='center', va='bottom', fontsize=fs, color='r')

    # Atmos temps
    ax.text(0.125*xmax, 0.75*ymax,
            r'$T_{A,S}$'+'$= {:.1f}$'.format(Ta[2])+r'$^\circ$C',
            ha='center', va='center', fontsize=fs,
            color=temp_cmap(temp_norm(Ta[2])))

    ax.text(0.5*xmax, 0.75*ymax,
            r'$T_{A,T}$'+'$= {:.1f}$'.format(Ta[0])+r'$^\circ$C',
            ha='center', va='center', fontsize=fs,
            color=temp_cmap(temp_norm(Ta[0])))

    ax.text(0.875*xmax, 0.75*ymax,
            r'$T_{A,N}$'+'$= {:.1f}$'.format(Ta[1])+r'$^\circ$C',
            ha='center', va='center', fontsize=fs,
            color=temp_cmap(temp_norm(Ta[1])))

    # Ocean temps and salinity
    ax.text(0.05*xmax + dx, 0.1*ymax,
            r'$T_{O,D}$'+'$= {:.1f}$'.format(To[2])+r'$^\circ$C',
            ha='left', va='top', fontsize=fs,
            color=temp_cmap(temp_norm(To[2])))
    ax.text(0.05*xmax + dx, 0.05*ymax,
            r'$S_{D}$'+'$= {:.1f}$ PSU'.format(S[2]),
            ha='left', va='top', fontsize=fs,
            color=salt_cmap(salt_norm(S[2])))

    ax.text(0.05*xmax + dx, 0.3*ymax,
            r'$T_{O,S}$'+'$= {:.1f}$'.format(To[3])+r'$^\circ$C',
            ha='left', va='top', fontsize=fs,
            color=temp_cmap(temp_norm(To[3])))
    ax.text(0.05*xmax + dx, 0.25*ymax,
            r'$S_{S}$'+'$= {:.1f}$ PSU'.format(S[3]),
            ha='left', va='top', fontsize=fs,
            color=salt_cmap(salt_norm(S[3])))

    ax.text(0.95*xmax - dx, 0.3*ymax,
            r'$T_{O,N}$'+'$= {:.1f}$'.format(To[1])+r'$^\circ$C',
            ha='right', va='top', fontsize=fs,
            color=temp_cmap(temp_norm(To[1])))
    ax.text(0.95*xmax - dx, 0.25*ymax,
            r'$S_{N}$'+'$= {:.1f}$ PSU'.format(S[1]),
            ha='right', va='top', fontsize=fs,
            color=salt_cmap(salt_norm(S[1])))

    ax.text(0.55*xmax, 0.4*ymax,
            r'$T_{O,T}$'+'$= {:.1f}$'.format(To[0])+r'$^\circ$C',
            ha='left', va='top', fontsize=fs,
            color=temp_cmap(temp_norm(To[0])))
    ax.text(0.55*xmax, 0.35*ymax,
            r'$S_{T}$'+'$= {:.1f}$ PSU'.format(S[0]),
            ha='left', va='top', fontsize=fs,
            color=salt_cmap(salt_norm(S[0])))

    ax.set_xlim(-dx, xmax+dx)
    ax.set_ylim(-dy, ymax+dy)
    ax.axis('off')
    return ax

