
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate(drawframe, N, filename, speedfactor=1, border=False, size=(1920, 1440), dpi=300, fps=30, loop=0):
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(size[0]/dpi, size[1]/dpi))
        
        if border:
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
        else:
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        n = N//speedfactor

        drawframe(ax, frame=0)
        def update(frame):
            ax.clear()
            index = frame*speedfactor
            drawframe(ax, frame=index)
            return None

        ani = animation.FuncAnimation(fig=fig, func=update, frames=n, interval=fps)

        writergif = animation.PillowWriter(fps = fps)
        writergif.setup(fig, filename, dpi = dpi) 
        ani.save(filename=filename, writer=writergif, loop=loop)