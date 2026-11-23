"""Editable conceptual MPC illustrations; all curves are explicitly schematic."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

CREDIT = 'Original course redraw informed by Drgoňa et al. (2020), Figs. 2–3 / Algorithm 1; simplified labels.'


def reference_schematic():
    """Draw the air/fabric heat-exchange paths in the Appendix A equations."""
    with plt.rc_context({'svg.fonttype':'none','font.size':18,'axes.grid':False}):
        fig,ax=plt.subplots(figsize=(40/3,7.5))
        fig.subplots_adjust(left=.025,right=.975,bottom=.13,top=.84)
        ax.set(xlim=(0,1),ylim=(0,1))
        ax.axis('off')
        fig.text(.04,.94,'Two-state reference building',fontsize=26,weight='bold')
        fig.text(.04,.885,'Heat exchange, thermal storage and the measured indoor temperature',fontsize=18)
        wire_colour='#40566a'

        # Outdoor-to-fabric path, corresponding to (To - Tf) / Rof.
        ax.plot([.10,.10,.40],[.50,.86,.86],color=wire_colour,lw=2)
        ax.plot([.54,.985,.985,.964],[.86,.86,.50,.50],color=wire_colour,lw=2)
        zigzag_x=np.linspace(.40,.54,9)
        zigzag_y=np.array([0,1,-1,1,-1,1,-1,1,0])*.018+.86
        ax.plot(zigzag_x,zigzag_y,color=wire_colour,lw=2)
        ax.text(.47,.94,r'$R_{of}=6.0$ K/kW',ha='center',fontsize=19)

        # Outdoor-to-air ventilation and air-to-fabric exchange.
        for left,right,label in [(.10,.34,r'$G_v=0.08$ kW/K'),
                                  (.58,.72,'$R_{af}$\n0.8 K/kW')]:
            middle=(left+right)/2
            half_width=.035
            ax.plot([left,middle-half_width],[.50,.50],color=wire_colour,lw=2)
            ax.plot([middle+half_width,right],[.50,.50],color=wire_colour,lw=2)
            x=np.linspace(middle-half_width,middle+half_width,9)
            y=np.array([0,1,-1,1,-1,1,-1,1,0])*.018+.50
            ax.plot(x,y,color=wire_colour,lw=2)
            ax.text(middle,.565,label,ha='center',va='bottom',fontsize=17)
        ax.scatter([.10],[.50],s=150,color=wire_colour,zorder=4)
        ax.text(.10,.40,r'Outdoor $T_o$',ha='center',fontsize=19)

        for x,colour,title,state,capacity in [
            (.34,'#e6f1fb','Indoor air',r'$T_a$',r'$C_a=0.8$ kWh/K'),
            (.72,'#fff0d9','Building fabric',r'$T_f$',r'$C_f=18.0$ kWh/K')]:
            ax.add_patch(FancyBboxPatch((x,.37),.24,.27,
                boxstyle='round,pad=.004',facecolor=colour,edgecolor=wire_colour,lw=1.8,zorder=3))
            ax.text(x+.12,.595,title,ha='center',va='center',fontsize=20,zorder=4)
            ax.text(x+.12,.505,state,ha='center',va='center',fontsize=26,zorder=4)
            ax.text(x+.12,.415,capacity,ha='center',va='center',fontsize=18,zorder=4)

        # The two heat-input arrows use the same allocation fractions as the ODEs.
        for x,label in [(.46,r'$\alpha_g Q_g+\alpha_h Q_{hp}$'),
                         (.84,r'$(1-\alpha_g)Q_g+(1-\alpha_h)Q_{hp}$')]:
            ax.text(x,.765,label,ha='center',va='center',fontsize=19)
            ax.annotate('',xy=(x,.645),xytext=(x,.72),
                arrowprops={'arrowstyle':'-|>','lw':2,'color':'#b35a13'})

        ax.add_patch(FancyBboxPatch((.29,.10),.34,.14,boxstyle='round,pad=.008',
            facecolor='#eef4ed',edgecolor='#476344',lw=1.5))
        ax.text(.46,.17,r'Sensor: $T^{meas}=T_a+\nu$',ha='center',va='center',fontsize=19)
        ax.annotate('',xy=(.46,.25),xytext=(.46,.36),
            arrowprops={'arrowstyle':'-|>','lw':1.8,'color':'#476344'})
        ax.text(.83,.15,r'$\alpha_g=0.45,\quad\alpha_h=0.25$',ha='center',fontsize=18)
        fig.text(.04,.065,r'$Q_g=A_s\max(I,0)+\max(Q_{int},0)$',fontsize=18)
        fig.text(.57,.065,r'$Q_{hp}=\mathrm{COP}_{ref}(T_o)P^{act}$',fontsize=18)
        fig.text(.04,.02,'Thermal schematic of the supplied synthetic reference; storage capacities are shown inside each state.',fontsize=13,color='0.3')
    return fig


def box(ax, x, y, w, h, text, colour='#eef3f8', size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.008',
                              facecolor=colour, edgecolor='#40566a'))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size)


def arrow(ax, start, end, text=None, pos=None):
    ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '->', 'lw': 1.6})
    if text:
        ax.text(*(pos or ((start[0]+end[0])/2, (start[1]+end[1])/2+.04)),
                text, ha='center', va='center', fontsize=10)


def canvas(title):
    fig, ax = plt.subplots(figsize=(12.5, 5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.set_title(title, fontsize=15, pad=12)
    return fig, ax


def loop(errors=False):
    fig, ax = canvas('Error entry points' if errors else 'MPC within building operation')
    if not errors:
        box(ax,.03,.79,.24,.13,'Supervisory MPC')
        box(ax,.39,.79,.24,.13,'BAS: local controllers / actuators')
        box(ax,.76,.79,.20,.13,'Building and sensors')
        arrow(ax,(.28,.855),(.38,.855),'Setpoints',(.33,.95))
        arrow(ax,(.64,.855),(.75,.855))
    box(ax,.02,.47,.21,.18,'Outdoor forecasts\nPrices and comfort')
    box(ax,.32,.43,.29,.26,'MPC\nThermal model + optimiser\nObjective and constraints')
    box(ax,.73,.48,.24,.17,'Building / reference model\nApplied electrical power')
    box(ax,.73,.12,.24,.15,'Sensor: available measurement')
    arrow(ax,(.24,.56),(.31,.56))
    arrow(ax,(.62,.56),(.72,.56),'P₀ only',(.67,.65))
    ax.text(.65,.34,'Plan: [P₀, P₁, …]\nLater actions provisional',ha='center',fontsize=10)
    arrow(ax,(.85,.47),(.85,.28))
    ax.plot([.72,.46,.46],[.195,.195,.42],color='black',lw=1.5)
    arrow(ax,(.46,.32),(.46,.42))
    ax.text(.43,.12,'New state anchor',ha='center',fontsize=11)
    if errors:
        ax.text(.11,.79,'2. Forecast error\nFuture input changes',ha='center',fontsize=11)
        arrow(ax,(.11,.74),(.11,.66))
        ax.text(.47,.83,'1. Model mismatch\nPrediction b,c stay fixed',ha='center',fontsize=11)
        arrow(ax,(.47,.76),(.47,.70))
        ax.text(.85,.87,'Process noise\nPhysical state changes',ha='center',fontsize=11)
        arrow(ax,(.85,.78),(.85,.66))
        ax.text(.85,.035,'3. Sensor error: measurement changes',ha='center',fontsize=10)
    else:
        ax.text(.13,.20,'Direct power commands\nare a teaching abstraction.',ha='center',fontsize=11)
    fig.text(.5,.015,CREDIT,ha='center',fontsize=8)
    return fig


def concept(figure_id):
    if figure_id == 'C81-01':return loop()
    if figure_id == 'C81-06':return loop(errors=True)
    if figure_id == 'C81-02':
        fig,ax=canvas('Signals and two distinct prediction tasks')
        table=[['State x / indoor T','Measured through sensor y / Tᵐᵉᵃˢ'],
               ['Control u / electrical P','Chosen by optimiser; kW'],
               ['Disturbance d / outdoor Tₒ, gains','External input; not chosen'],
               ['Price p / comfort L,U','Announced objective / limits'],
               ['Parameters θ / b,c','Supplied, fixed in the core'],
               ['Reference fabric temperature','Hidden from controller']]
        tab=ax.table(cellText=table,colWidths=[.44,.56],bbox=[.02,.52,.96,.46],cellLoc='left')
        tab.auto_set_font_size(False);tab.set_fontsize(11)
        box(ax,.02,.16,.42,.22,'Weather forecaster → future outdoor Tₒ\nHere: supplied synthetic forecast table')
        box(ax,.55,.16,.43,.22,'Thermal predictor → future indoor T\nInputs: current Tᵐᵉᵃˢ, candidate P, forecast Tₒ')
        ax.text(.5,.04,'Price and comfort enter optimisation separately; the optimiser does not choose weather.',ha='center',fontsize=11)
        return fig
    if figure_id == 'C81-03':
        fig,axes=plt.subplots(2,3,figsize=(12.5,5.5),sharex=True,sharey='row')
        # Arbitrary illustration only: no empirical quantities or uncertainty intervals.
        for k in range(3):
            upper,lower=axes[:,k];now=6*k;end=now+24
            upper.axhspan(20,23,color='#dcefdc',label='Comfort band (schematic)')
            upper.plot([-6,0,6,12][:k+2],[21,20.8,20.6,20.9][:k+2],color='black',label='Past measurements')
            anchor=[20.8,20.6,20.9][k]
            upper.plot([now,now+6,now+12,end],[anchor,21.5,20.5,21.1],ls='--',color='tab:blue',label='Predicted T')
            upper.scatter([now],[anchor],color='black',zorder=5,label='New observation')
            lower.stairs([2,1,3,1],[now,now+.5,now+6,now+12,end],ls='--',color='tab:blue',label='Planned P')
            lower.plot([now,now+.5],[2,2],color='tab:orange',lw=6,label='First applied action')
            for ax in (upper,lower):
                ax.axvline(now,color='0.4',ls=':');ax.axvspan(now,now+.5,facecolor='none',edgecolor='tab:orange',hatch='///')
                ax.set_xlim(-7,37)
            upper.set_ylim(19,24);lower.set_ylim(0,4)
            upper.set_title(f'Illustrative origin {now:g} h; end {end:g} h',fontsize=11)
            upper.annotate('',xy=(now,23.5),xytext=(end,23.5),arrowprops={'arrowstyle':'<->'})
            upper.text(now+12,23.65,'NΔt = 24 h',ha='center',fontsize=9)
            lower.set_xlabel('Hours from first origin')
        axes[0,0].set_ylabel('Indoor T (schematic)');axes[1,0].set_ylabel('Heating P (schematic)')
        axes[0,0].legend(fontsize=7,loc='lower right');axes[1,0].legend(fontsize=8,loc='upper right')
        fig.subplots_adjust(bottom=.27,top=.88,hspace=.28,wspace=.14)
        inset=fig.add_axes([.29,.035,.48,.12]);inset.set(xlim=(0,25),ylim=(-.5,2.5));inset.axis('off')
        for k in range(3):inset.plot([k*.5,24],[2-k]*2,'o-',color='0.4',ms=3)
        inset.text(-.3,1,'Shrinking horizon:',ha='right',va='center',fontsize=9)
        inset.text(24.4,1,'Fixed endpoint',va='center',fontsize=9)
        fig.suptitle('Moving horizon: illustrative origins 6 h apart; core updates every 30 min',fontsize=13)
        fig.text(.5,.005,CREDIT,ha='center',fontsize=7)
        return fig
    if figure_id == 'C81-04':
        fig,ax=canvas('The six-step MPC algorithm')
        labels=['1. Read measurement\nview = session.available()',
                '2. Gather inputs\ninputs = session.inputs(...)',
                '3. Optimise and check\nupdate Parameters;\nsolve_update(...)',
                '4. Select action\nfirst_action(plan)',
                '5. Apply and observe\nnext_measurement =\nsession.advance(...)',
                '6. Repeat\nnext iteration; same horizon N']
        for i,label in enumerate(labels):
            x=.03+(i%3)*.33;y=.64 if i<3 else .27
            box(ax,x,y,.28,.21,label,size=10)
        arrow(ax,(.32,.745),(.35,.745));arrow(ax,(.65,.745),(.68,.745))
        ax.plot([.96,.99,.99,.01,.01],[.64,.59,.54,.54,.375],color='black',lw=1)
        arrow(ax,(.01,.375),(.03,.375))
        arrow(ax,(.32,.375),(.35,.375));arrow(ax,(.65,.375),(.68,.375))
        ax.plot([.82,.82,.17],[.26,.16,.16],color='black',lw=1)
        # Explicit return to measurement, routed outside the boxes.
        ax.plot([.17,.005,.005],[.16,.16,.90],color='black',lw=1)
        arrow(ax,(.005,.90),(.17,.85))
        ax.text(.5,.065,'Updated: measurement, forecasts, plan     |     Fixed in the core: b,c',ha='center',fontsize=12)
        return fig
    if figure_id == 'C81-05':
        fig,ax=canvas('Prediction model and reference model have different roles')
        box(ax,.03,.45,.40,.38,'INSIDE CONTROLLER\nOne-state RC prediction\nCandidate P + available inputs → predicted T\nFixed b,c; current measured anchor',size=12)
        box(ax,.57,.45,.40,.38,'OUTSIDE CONTROLLER\nDay 1: richer air/fabric reference\nApplied P + realised inputs → true state\nGains, variable COP and noise',size=12)
        arrow(ax,(.44,.64),(.56,.64),'First P only',(.5,.72))
        box(ax,.57,.12,.40,.17,'Sensor → available indoor measurement',size=12)
        arrow(ax,(.77,.44),(.77,.30))
        ax.plot([.56,.23,.23],[.205,.205,.44],color='black',lw=1.5)
        arrow(ax,(.23,.35),(.23,.44))
        ax.text(.23,.10,'Truth / fabric state: evaluator only',ha='center',fontsize=10)
        ax.text(.5,.02,'Day 2 controlled experiments use a separately labelled one-state reference.',ha='center',fontsize=10)
        return fig
    raise ValueError(figure_id)
